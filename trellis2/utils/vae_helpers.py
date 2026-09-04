from __future__ import annotations

import gc
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple
import matplotlib.pyplot as plt
from collections import deque, defaultdict

import numpy as np
import torch
import torch.nn.functional as F


# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import pyvista as pv


# Helper for Python < 3.10 compatibility
class nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


# ---------------------------------------------------------------------
# Environment tweaks
# ---------------------------------------------------------------------
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["ATTN_BACKEND"] = "xformers"

# Check if we are in a notebook
try:
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.modules.sparse import SparseTensor
    import trellis2.models as models
except ImportError:
    print(
        "Error: trellis2 module not found. Make sure you are in the correct environment."
    )

from PIL import Image

os.environ["PYOPENGL_PLATFORM"] = (
    "egl"  # must come before importing pyrender, OpenGL, etc.
)

# --- pyrender imports must come after pyglet option set ---
import pyglet

pyglet.options["shadow_window"] = False


# ---------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------
# Reset handlers to avoid duplicate logs if cell is re-run
logging.root.handlers = []
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
LOGGER = logging.getLogger("decode_shapes")


def log_cuda_memory(tag: str) -> None:
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    LOGGER.info(
        f"[{tag}] CUDA alloc={alloc:.2f}GB reserved={reserved:.2f}GB peak={peak:.2f}GB"
    )


def cleanup_cuda(*tensors) -> None:
    """Explicitly delete tensors and clear cache."""
    for t in tensors:
        try:
            del t
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def add_batch_dim_if_missing(coords: torch.Tensor) -> torch.Tensor:
    # Expected SparseTensor coords are often [N,4] = [b,x,y,z]
    if coords.ndim != 2:
        raise ValueError(f"coords must be 2D, got {coords.shape}")
    if coords.shape[1] == 3:
        zeros = torch.zeros(
            (coords.shape[0], 1), dtype=torch.int32, device=coords.device
        )
        coords = torch.cat([zeros, coords.to(torch.int32)], dim=1)
    return coords.to(torch.int32).contiguous()


def load_shape_latent_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        feats = data["feats"]
        coords = data["coords"]
    return feats, coords


def load_coords_latent_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        coords = data["coords"]
    return coords


def load_ss_latent_npz(path: Path) -> np.ndarray:
    with np.load(path) as data:
        z = data["z"]
    return z


# ---------------------------------------------------------------------
# 3D helpers
# ---------------------------------------------------------------------


def get_concat_h(im1, im2):
    dst = Image.new("RGB", (im1.width + im2.width, max(im1.height, im2.height)))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def coords_to_xyz(coords: torch.Tensor) -> np.ndarray:
    """
    coords: [N,4] int tensor with columns [b,x,y,z] or [N,3] [x,y,z]
    returns: [N,3] float numpy array [x,y,z]
    """
    if coords.shape[1] == 4:
        xyz = coords[:, 1:4]
    elif coords.shape[1] == 3:
        xyz = coords
    else:
        raise ValueError(f"coords must have shape [N,3] or [N,4], got {coords.shape}")
    return xyz.detach().cpu().numpy().astype(np.float32)


def nearest_neighbor_kdtree(
    coords_a: torch.Tensor, coords_b: torch.Tensor, batch_id: int = 0, k: int = 1
):
    """
    Returns:
      nn_idx: (Na,) indices into B for nearest neighbor of each A
      nn_dist: (Na,) Euclidean distance
    """
    from scipy.spatial import cKDTree

    # Filter by batch if needed
    if coords_a.shape[1] == 4:
        coords_a = coords_a[coords_a[:, 0] == batch_id][:, 1:4]
    if coords_b.shape[1] == 4:
        coords_b = coords_b[coords_b[:, 0] == batch_id][:, 1:4]

    a = coords_a.detach().cpu().numpy().astype(np.float32)
    b = coords_b.detach().cpu().numpy().astype(np.float32)

    tree = cKDTree(b)
    nn_dist, nn_idx = tree.query(a, k=k, workers=-1)  # workers=-1 uses all cores

    return torch.from_numpy(nn_idx), torch.from_numpy(nn_dist)


def partition_voxels(
    coords,
    max_chunk_size,
    max_radius,
    min_chunk_size=1,
):
    """
    Partition an (N, 3) integer voxel array into 6-connected chunks,
    capping each chunk by both size and BFS graph radius, then merging
    tiny chunks into neighboring chunks.

    Returns indices into the ORIGINAL coords array instead of coordinates.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 3)
        Integer voxel coordinates.
    max_chunk_size : int
        Soft maximum number of UNIQUE voxel positions per chunk during initial BFS.
    max_radius : int
        Maximum BFS graph distance from the seed voxel during initial growth.
    min_chunk_size : int
        Minimum desired size of final chunks, measured in UNIQUE voxel positions.

    Returns
    -------
    chunks_idx : list[np.ndarray]
        List of 1D integer arrays. Each array contains row indices into the
        original `coords` array belonging to that chunk.

    Notes
    -----
    - Connectivity is computed on UNIQUE voxel coordinates.
    - If the same coordinate appears multiple times in `coords`, all of its
      original indices are included in the final chunk.
    """
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3)")
    if not np.issubdtype(coords.dtype, np.integer):
        raise ValueError("coords must contain integers")
    if max_chunk_size < 1:
        raise ValueError("max_chunk_size must be >= 1")
    if max_radius < 0:
        raise ValueError("max_radius must be >= 0")
    if min_chunk_size < 1:
        raise ValueError("min_chunk_size must be >= 1")

    # Map each voxel coordinate -> all original row indices having that coordinate
    coord_to_indices = defaultdict(list)
    for i, v in enumerate(coords):
        coord_to_indices[tuple(map(int, v))].append(i)

    # Unique voxel coordinates only
    unique_voxels = sorted(coord_to_indices.keys())
    remaining = set(unique_voxels)

    neighbors = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]

    # --------------------------------------------------
    # Phase 1: initial BFS chunking on UNIQUE coordinates
    # --------------------------------------------------
    chunks = []  # each chunk is a set of voxel tuples

    for seed in unique_voxels:
        if seed not in remaining:
            continue

        queue = deque([(seed, 0)])  # (voxel, bfs_distance_from_seed)
        queued = {seed}
        chunk = []

        while queue and len(chunk) < max_chunk_size:
            voxel, dist = queue.popleft()

            if voxel not in remaining:
                continue
            if dist > max_radius:
                continue

            remaining.remove(voxel)
            chunk.append(voxel)

            if dist == max_radius:
                continue

            x, y, z = voxel
            for dx, dy, dz in neighbors:
                nb = (x + dx, y + dy, z + dz)
                if nb in remaining and nb not in queued:
                    queue.append((nb, dist + 1))
                    queued.add(nb)

        chunks.append(set(chunk))

    # --------------------------------------------------
    # Phase 2: merge chunks smaller than min_chunk_size
    # --------------------------------------------------
    def chunk_centroid(chunk):
        arr = np.array(list(chunk), dtype=float)
        return arr.mean(axis=0)

    def build_voxel_to_chunk(chunks):
        voxel_to_chunk = {}
        for ci, chunk in enumerate(chunks):
            for v in chunk:
                voxel_to_chunk[v] = ci
        return voxel_to_chunk

    def find_adjacent_chunks(chunk_idx, chunks, voxel_to_chunk):
        contacts = {}
        for x, y, z in chunks[chunk_idx]:
            for dx, dy, dz in neighbors:
                nb = (x + dx, y + dy, z + dz)
                other = voxel_to_chunk.get(nb, None)
                if other is not None and other != chunk_idx:
                    contacts[other] = contacts.get(other, 0) + 1
        return contacts

    def find_nearest_chunk(chunk_idx, chunks):
        c0 = chunk_centroid(chunks[chunk_idx])
        best_j = None
        best_d2 = None
        for j, chunk in enumerate(chunks):
            if j == chunk_idx:
                continue
            c1 = chunk_centroid(chunk)
            d2 = np.sum((c0 - c1) ** 2)
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_j = j
        return best_j

    changed = True
    while changed:
        changed = False
        voxel_to_chunk = build_voxel_to_chunk(chunks)

        small_indices = [i for i, ch in enumerate(chunks) if len(ch) < min_chunk_size]
        if not small_indices:
            break

        small_indices.sort(key=lambda i: len(chunks[i]))

        for i in small_indices:
            if i >= len(chunks):
                continue
            if len(chunks[i]) >= min_chunk_size:
                continue
            if len(chunks) == 1:
                break

            adjacent = find_adjacent_chunks(i, chunks, voxel_to_chunk)

            if adjacent:
                target = max(
                    adjacent.items(), key=lambda kv: (kv[1], len(chunks[kv[0]]))
                )[0]
            else:
                target = find_nearest_chunk(i, chunks)

            if target is None:
                continue

            chunks[target].update(chunks[i])
            del chunks[i]
            changed = True
            break

    # --------------------------------------------------
    # Phase 3: expand each unique-voxel chunk back to original row indices
    # --------------------------------------------------
    chunks_idx = []
    for chunk in chunks:
        idx = []
        for voxel in chunk:
            idx.extend(coord_to_indices[voxel])
        chunks_idx.append(np.array(sorted(idx), dtype=np.int64))

    return chunks_idx


# ---------------------------------------------------------------------
# Decoding helpers
# ---------------------------------------------------------------------


def decode_sparse_structure_coords(
    decoder,
    z_s: torch.Tensor,
    target_resolution: int = 64,
    pool_on_cpu: bool = True,
) -> torch.Tensor:
    """
    Decodes sparse structure latent z_s into coordinates.
    """

    if decoder.low_vram:
        decoder.to(decoder.device)

    decoded = decoder(z_s)

    decoded = decoded > 0  # bool tensor

    if decoder.low_vram:
        decoder.cpu()

    LOGGER.info(f"Decoded sparse structure shape before max pool: {decoded.shape}")

    if decoded.shape[2] != target_resolution:
        ratio = decoded.shape[2] // target_resolution
        if pool_on_cpu:
            decoded_cpu = decoded.to("cpu")
            pooled = F.max_pool3d(decoded_cpu.to(torch.float16), ratio, ratio, 0) > 0.5
            decoded = pooled  # stays on CPU
        else:
            decoded = F.max_pool3d(decoded.to(torch.float16), ratio, ratio, 0) > 0.5

    coords = (
        decoded.nonzero(as_tuple=False)[:, [0, 2, 3, 4]].to(torch.int32).contiguous()
    )

    return coords, decoded


def decode_meshes_from_shape_slat(
    shape_decoder,
    shape_slat: SparseTensor,
    resolution: int,
    return_subs: bool = False,
):
    """
    Decodes shape features into a mesh.
    """
    shape_decoder.set_resolution(resolution)

    if shape_decoder.low_vram:
        shape_decoder.to(shape_decoder.device)
        shape_decoder.low_vram = True

    ret = shape_decoder(shape_slat, return_subs=return_subs)

    if shape_decoder.low_vram:
        shape_decoder.cpu()
        shape_decoder.low_vram = False

    if return_subs:
        meshes, subs = ret
        return meshes, subs

    if isinstance(ret, tuple) and len(ret) == 2:
        meshes, _subs = ret
        return meshes

    return ret


# ---------------------------------------------------------------------
# Visualization and rendering helpers
# ---------------------------------------------------------------------


def plot_voxels_scatter(
    coords: torch.Tensor, voxel_size=1.0, origin=(0, 0, 0), max_points=200_000
):
    xyz = coords_to_xyz(coords)

    # Convert indices -> world coordinates (voxel centers)
    origin = np.array(origin, dtype=np.float32)
    xyz_world = origin + (xyz + 0.5) * float(voxel_size)

    # Downsample if huge
    if xyz_world.shape[0] > max_points:
        idx = np.random.choice(xyz_world.shape[0], size=max_points, replace=False)
        xyz_world = xyz_world[idx]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz_world[:, 0], xyz_world[:, 1], xyz_world[:, 2], s=10)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Rough equal aspect
    mins = xyz_world.min(axis=0)
    maxs = xyz_world.max(axis=0)
    spans = maxs - mins
    center = (mins + maxs) / 2
    radius = spans.max() / 2
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    plt.tight_layout()
    plt.show()


def plot_voxel_slice(coords: torch.Tensor, axis="z", index=None):
    """
    Shows a 2D occupancy slice (great for sanity checks).
    axis: 'x'|'y'|'z'
    index: slice coordinate; if None, uses median.
    """
    xyz = coords_to_xyz(coords).astype(np.int32)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    if axis == "x":
        if index is None:
            index = int(np.median(x))
        mask = x == index
        a, b = y[mask], z[mask]
        a_name, b_name = "y", "z"
    elif axis == "y":
        if index is None:
            index = int(np.median(y))
        mask = y == index
        a, b = x[mask], z[mask]
        a_name, b_name = "x", "z"
    elif axis == "z":
        if index is None:
            index = int(np.median(z))
        mask = z == index
        a, b = x[mask], y[mask]
        a_name, b_name = "x", "y"
    else:
        raise ValueError("axis must be one of 'x','y','z'")

    if mask.sum() == 0:
        print(f"No voxels in {axis}={index}")
        return

    a_min, a_max = a.min(), a.max()
    b_min, b_max = b.min(), b.max()
    img = np.zeros((b_max - b_min + 1, a_max - a_min + 1), dtype=np.uint8)
    img[b - b_min, a - a_min] = 255

    plt.figure()
    plt.imshow(img, origin="lower", interpolation="nearest")
    plt.title(f"Occupancy slice {axis}={index} ({a_name} vs {b_name})")
    plt.xlabel(a_name)
    plt.ylabel(b_name)
    plt.show()


def export_voxels_as_cubes_mesh(
    out_path: str,
    coords: torch.Tensor,
    voxel_size=1.0,
    origin=(0, 0, 0),
    max_voxels=500_000,
):
    import open3d as o3d

    if coords.shape[1] == 4:
        xyz = coords[:, 1:4]
    else:
        xyz = coords

    xyz = xyz.detach().cpu().numpy().astype(np.float32)
    origin = np.array(origin, dtype=np.float32)

    if xyz.shape[0] > max_voxels:
        idx = np.random.choice(xyz.shape[0], size=max_voxels, replace=False)
        xyz = xyz[idx]
        print(f"Downsampled to {max_voxels} voxels for cube-mesh export.")

    mesh_all = o3d.geometry.TriangleMesh()
    for v in xyz:
        # cube corner at voxel origin (not center)
        corner = origin + v * float(voxel_size)
        cube = o3d.geometry.TriangleMesh.create_box(
            width=voxel_size, height=voxel_size, depth=voxel_size
        )
        cube.translate(corner)
        mesh_all += cube

    mesh_all.compute_vertex_normals()
    o3d.io.write_triangle_mesh(out_path, mesh_all)
    print(f"Wrote cube-mesh: {out_path}")


def write_ply_binary(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """
    Writes a binary_little_endian PLY with vertex positions and triangular faces.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    v = np.asarray(vertices, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int32)

    with path.open("wb") as f_out:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {v.shape[0]}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            f"element face {f.shape[0]}\n"
            "property list int int vertex_indices\n"
            "end_header\n"
        )
        f_out.write(header.encode("ascii"))
        f_out.write(v.tobytes())

        counts = np.full((f.shape[0], 1), 3, dtype=np.int32)
        faces_data = np.hstack([counts, f])
        f_out.write(faces_data.tobytes())


def explode(data):
    size = np.array(data.shape) * 2
    data_e = np.zeros(size - 1, dtype=data.dtype)
    data_e[::2, ::2, ::2] = data
    return data_e


def render_voxels_mpl_toolkits(coords, out_path):
    """ """
    # Shift to a compact index space for the boolean occupancy grid

    if coords.shape[1] == 4:
        coords = coords[:, 1:4]

    mins = coords.min(axis=0)
    idx = coords - mins
    shape = idx.max(axis=0) + 1

    filled = np.zeros(shape, dtype=bool)
    filled[tuple(idx.T)] = True

    facecolors = np.where(filled, "#FFD75D7E", "#7A88CC77")
    edgecolors = np.where(filled, "#BFAB6E", "#7D84A6")

    # upscale the above voxel image, leaving gaps
    filled_2 = explode(filled)
    fcolors_2 = explode(facecolors)
    ecolors_2 = explode(edgecolors)

    # Shrink the gaps
    x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2
    x[0::2, :, :] += 0.05
    y[:, 0::2, :] += 0.05
    z[:, :, 0::2] += 0.05
    x[1::2, :, :] += 0.95
    y[:, 1::2, :] += 0.95
    z[:, :, 1::2] += 0.95

    fig = plt.figure()
    # ax = fig.gca(projection="3d")
    ax = fig.add_subplot(projection="3d")
    ax.voxels(x, y, z, filled_2, facecolors=fcolors_2, edgecolors=ecolors_2)
    # ax.voxels(filled_2, facecolors=fcolors_2, edgecolors=ecolors_2)
    # ax.voxels(filled, facecolors=facecolors, edgecolors=edgecolors)
    ax.set_axis_off()
    plt.show()

    plt.savefig(out_path, dpi=300)


def render_voxels_pyvista(
    coords,
    surf_idx=None,
    bound_idx=None,
    out_path=None,
    voxel_size=1.0,
    opacity=0.55,
    surf_color=(10, 215, 23),
    bound_color=(23, 10, 215),
    edge_color="black",
    background="white",
    window_size=(1024, 1024),
    as_pil=True,
):

    if coords.shape[1] == 4:
        coords = coords[:, 1:4]

    coords = np.asarray(coords, dtype=np.int32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3)")

    mins = coords.min(axis=0)
    ijk = coords - mins
    shape = ijk.max(axis=0) + 1  # voxel counts in x,y,z

    # Build the uniform grid
    grid = pv.ImageData(
        dimensions=tuple(shape + 1),  # cells = shape
        spacing=(voxel_size, voxel_size, voxel_size),
        origin=tuple(mins * voxel_size),
    )

    # Flat cell ids for the occupied voxels
    occ_flat = np.ravel_multi_index(ijk.T, dims=tuple(shape), order="F")

    # RGBA color for every cell in the full grid
    n_cells_total = int(np.prod(shape))
    alpha = int(round(255 * opacity))
    colors = np.zeros((n_cells_total, 4), dtype=np.uint8)  # transparent by default

    # Default color for all occupied voxels

    # Override subsets
    if surf_idx is not None:
        colors[occ_flat[surf_idx]] = (*surf_color, alpha)
    if bound_idx is not None:
        colors[occ_flat[bound_idx]] = (*bound_color, alpha)

    # Attach colors to the grid's cells, then extract only occupied cells
    grid.cell_data["colors"] = colors
    voxels = grid.extract_cells(occ_flat)

    # Render
    pl = pv.Plotter(off_screen=True, window_size=window_size)
    pl.set_background(background)
    pl.add_mesh(
        voxels,
        scalars="colors",
        rgb=True,
        show_edges=True,
        edge_color=edge_color,
        line_width=1.0,
        smooth_shading=False,
        lighting=True,
        show_scalar_bar=False,
    )
    pl.view_isometric()
    pl.show(auto_close=False)

    if out_path is not None:
        pl.screenshot(out_path)
        pl.close()
        return None
    else:
        img = pl.screenshot(filename=None, return_img=True)
        pl.close()
        return Image.fromarray(img) if as_pil else img


def render_voxels_pyvista_video(
    coords,
    surf_idx=None,
    bound_idx=None,
    out_path=None,
    voxel_size=1.0,
    opacity=0.55,
    surf_color=(10, 215, 23),
    bound_color=(23, 10, 215),
    edge_color="black",
    background="white",
    window_size=(1024, 1024),
    as_pil=True,
    n_frames=60,  # New: Number of frames in the video
    fps=15,  # New: Frames per second
):

    if coords.shape[1] == 4:
        coords = coords[:, 1:4]

    coords = np.asarray(coords, dtype=np.int32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3)")

    mins = coords.min(axis=0)
    ijk = coords - mins
    shape = ijk.max(axis=0) + 1  # voxel counts in x,y,z

    # Build the uniform grid
    grid = pv.ImageData(
        dimensions=tuple(shape + 1),  # cells = shape
        spacing=(voxel_size, voxel_size, voxel_size),
        origin=tuple(mins * voxel_size),
    )

    # Flat cell ids for the occupied voxels
    occ_flat = np.ravel_multi_index(ijk.T, dims=tuple(shape), order="F")

    # RGBA color for every cell in the full grid
    n_cells_total = int(np.prod(shape))
    alpha = int(round(255 * opacity))
    colors = np.zeros((n_cells_total, 4), dtype=np.uint8)  # transparent by default

    # Override subsets
    if surf_idx is not None:
        colors[occ_flat[surf_idx]] = (*surf_color, alpha)
    if bound_idx is not None:
        colors[occ_flat[bound_idx]] = (*bound_color, alpha)

    # Attach colors to the grid's cells, then extract only occupied cells
    grid.cell_data["colors"] = colors
    voxels = grid.extract_cells(occ_flat)

    # Render Setup
    pl = pv.Plotter(off_screen=True, window_size=window_size)
    pl.set_background(background)
    pl.add_mesh(
        voxels,
        scalars="colors",
        rgb=True,
        show_edges=True,
        edge_color=edge_color,
        line_width=1.0,
        smooth_shading=False,
        lighting=True,
        show_scalar_bar=False,
    )

    pl.view_isometric()
    pl.show(auto_close=False)

    # Video / Multi-frame setup
    save_file = out_path is not None
    if save_file:
        if out_path.lower().endswith(".gif"):
            pl.open_gif(out_path, fps=fps)
        else:
            pl.open_movie(out_path, framerate=fps)

    frames = []
    angle_step = 360.0 / n_frames

    # Render loop
    for i in range(n_frames):
        # Rotate camera for every frame after the first one
        if i > 0:
            pl.camera.azimuth += angle_step

        pl.render()

        if save_file:
            pl.write_frame()
        else:
            # Collect frames in memory if no out_path is provided
            img = pl.screenshot(filename=None, return_img=True)
            frames.append(Image.fromarray(img) if as_pil else img)

    pl.close()

    # Return None if saved to file, otherwise return the list of frame images
    if save_file:
        return None
    return frames
