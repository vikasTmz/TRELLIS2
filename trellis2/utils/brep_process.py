import sys
import random
import math
import re
import os
import numpy as np
import trimesh
from collections import defaultdict
from scipy.spatial import cKDTree
from skimage import measure
from tqdm.auto import tqdm


# --- Vector Math Helpers ---
def vec_sub(v1, v2):
    return [v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]]


def vec_len(v):
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def cross_product(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def triangle_area(v0, v1, v2):
    edge1 = vec_sub(v1, v0)
    edge2 = vec_sub(v2, v0)
    cross = cross_product(edge1, edge2)
    return 0.5 * vec_len(cross)


def line_length(v0, v1):
    return vec_len(vec_sub(v1, v0))


# --- Sampling Helpers ---
def get_random_point_on_triangle(v0, v1, v2):
    r1 = random.random()
    r2 = random.random()
    sqrt_r1 = math.sqrt(r1)

    w0 = 1.0 - sqrt_r1
    w1 = sqrt_r1 * (1.0 - r2)
    w2 = sqrt_r1 * r2

    return [
        w0 * v0[0] + w1 * v1[0] + w2 * v2[0],
        w0 * v0[1] + w1 * v1[1] + w2 * v2[1],
        w0 * v0[2] + w1 * v1[2] + w2 * v2[2],
    ]


def get_random_point_on_line(v0, v1):
    t = random.random()
    return [
        (1.0 - t) * v0[0] + t * v1[0],
        (1.0 - t) * v0[1] + t * v1[1],
        (1.0 - t) * v0[2] + t * v1[2],
    ]


def random_unique_rgb(N, seed=None, dtype=np.uint8):
    """
    Returns an (N, 3) array of unique random RGB colors.

    dtype:
      - np.uint8  -> values in [0, 255]
      - np.float32/float64 -> values in [0.0, 1.0]
    """
    max_colors = 256**3
    if N > max_colors:
        raise ValueError(f"N must be <= {max_colors} (got {N})")

    rng = np.random.default_rng(seed)

    # Sample unique colors by sampling unique integers in [0, 256^3)
    idx = rng.choice(max_colors, size=N, replace=False, shuffle=False).astype(np.uint32)

    # Convert integer -> (R, G, B) via base-256 decomposition
    rgb = np.empty((N, 3), dtype=np.uint8)
    rgb[:, 0] = (idx // (256 * 256)) & 255  # R
    rgb[:, 1] = (idx // 256) & 255  # G
    rgb[:, 2] = idx & 255  # B

    if np.issubdtype(dtype, np.floating):
        return rgb.astype(dtype) / 255.0
    return rgb.astype(dtype)


def build_smooth_union_sdf(points, xs, ys, zs, radius, k, eps=1e-6):
    # Influence radius where exp(-k*di) < eps (near the surface di~0)
    band = -np.log(eps) / k
    R = radius + band

    nx, ny, nz = len(xs), len(ys), len(zs)

    # Running min and scaled exp-sum accumulator
    m = np.full((nx, ny, nz), np.float32(1e6), dtype=np.float32)  # big positive
    s = np.zeros((nx, ny, nz), dtype=np.float32)

    xs_f = xs.astype(np.float32, copy=False)
    ys_f = ys.astype(np.float32, copy=False)
    zs_f = zs.astype(np.float32, copy=False)
    radius = np.float32(radius)
    k = np.float32(k)

    for px, py, pz in tqdm(
        points.astype(np.float32, copy=False),
        total=len(points),
        desc="Blending spheres",
        unit="sphere",
        bar_format="{l_bar}{bar} {n_fmt}/{total_fmt} "
        "[elapsed: {elapsed} | remaining: {remaining} | {rate_fmt}{postfix}]",
    ):
        # for px, py, pz in points.astype(np.float32, copy=False):
        # index ranges for |coord - p| <= R
        ix0 = np.searchsorted(xs_f, px - R, side="left")
        ix1 = np.searchsorted(xs_f, px + R, side="right")
        iy0 = np.searchsorted(ys_f, py - R, side="left")
        iy1 = np.searchsorted(ys_f, py + R, side="right")
        iz0 = np.searchsorted(zs_f, pz - R, side="left")
        iz1 = np.searchsorted(zs_f, pz + R, side="right")

        if ix0 >= ix1 or iy0 >= iy1 or iz0 >= iz1:
            continue

        dx2 = (xs_f[ix0:ix1] - px) ** 2
        dy2 = (ys_f[iy0:iy1] - py) ** 2
        dz2 = (zs_f[iz0:iz1] - pz) ** 2

        di = (
            np.sqrt(dx2[:, None, None] + dy2[None, :, None] + dz2[None, None, :])
            - radius
        )

        m0 = m[ix0:ix1, iy0:iy1, iz0:iz1]
        s0 = s[ix0:ix1, iy0:iy1, iz0:iz1]

        new_m = np.minimum(m0, di)

        # rescale old sum to new_m (avoid exp(inf) etc.)
        # when m0 was the big default and s0==0, this is fine.
        scale = np.exp(-k * (m0 - new_m)).astype(np.float32, copy=False)
        s_new = s0 * scale + np.exp(-k * (di - new_m)).astype(np.float32, copy=False)

        m0[...] = new_m
        s0[...] = s_new

    # finalize: d = m - log(s)/k ; where s==0, keep big positive
    out = m.copy()
    mask = s > 0
    out[mask] = m[mask] - (1.0 / k) * np.log(s[mask])

    return out


def smooth_min(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """
    Smooth minimum via log-sum-exp:
        smin(a,b) = -(1/k) * log(exp(-k*a) + exp(-k*b))
    """
    # Numerical stability: factor out min
    m = np.minimum(a, b)
    return m - (1.0 / k) * np.log(np.exp(-k * (a - m)) + np.exp(-k * (b - m)))


def points_to_metaball_mesh(
    points: np.ndarray,
    radius: float = 0.2,
    smooth_k: float | None = None,
    voxel_size: float = 0.02,
    padding: float = 0.4,
) -> trimesh.Trimesh:
    """
    Convert 3D points to a blended metaball/smooth-union surface mesh.

    points: (N,3) float array
    radius: sphere radius in same units as points
    smooth_k: smoothness; if None, uses ~8/radius (reasonable default)
    voxel_size: grid spacing; smaller -> more detail, slower, more memory
    padding: extra margin around the point cloud bounds
    """
    print("Convert 3D points to a blended metaball/smooth-union surface mesh...")

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be shape (N, 3)")
    if len(points) == 0:
        raise ValueError("points array is empty")

    if smooth_k is None:
        smooth_k = 8.0 / max(radius, 1e-6)

    # Bounding box (+ padding and radius margin)
    pmin = points.min(axis=0) - (padding + 2.0 * radius)
    pmax = points.max(axis=0) + (padding + 2.0 * radius)

    # Grid dimensions
    dims = np.ceil((pmax - pmin) / voxel_size).astype(int) + 1
    nx, ny, nz = dims.tolist()

    # Grid coordinates (world space)
    xs = np.linspace(pmin[0], pmax[0], nx, dtype=np.float32)
    ys = np.linspace(pmin[1], pmax[1], ny, dtype=np.float32)
    zs = np.linspace(pmin[2], pmax[2], nz, dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    print("Build smooth-union SDF over the grid")
    print(f"Processing {points.shape[0]} points")
    # d = None
    # for px, py, pz in tqdm(
    #     points,
    #     total=len(points),
    #     desc="Blending spheres",
    #     unit="sphere",
    #     bar_format="{l_bar}{bar} {n_fmt}/{total_fmt} "
    #     "[elapsed: {elapsed} | remaining: {remaining} | {rate_fmt}{postfix}]",
    # ):
    #     di = np.sqrt((X - px) ** 2 + (Y - py) ** 2 + (Z - pz) ** 2) - radius
    #     d = di if d is None else smooth_min(d, di, smooth_k)

    d = build_smooth_union_sdf(points, xs, ys, zs, radius=radius, k=smooth_k, eps=1e-6)

    # Marching cubes: surface is where SDF == 0
    dx = xs[1] - xs[0] if nx > 1 else 1.0
    dy = ys[1] - ys[0] if ny > 1 else 1.0
    dz = zs[1] - zs[0] if nz > 1 else 1.0

    print("Running marching cubes...")

    verts, faces, norms, _ = measure.marching_cubes(
        volume=d,
        level=0.0,
        spacing=(dx, dy, dz),
    )

    # marching_cubes outputs vertices in grid-space starting at (0,0,0) -> add origin
    verts = verts + pmin

    mesh = trimesh.Trimesh(
        vertices=verts, faces=faces, vertex_normals=norms, process=True
    )

    return mesh


def export_mesh(mesh: trimesh.Trimesh, out_ply: str, out_glb: str) -> None:
    mesh.export(out_ply)  # PLY
    # GLB export: wrap in a scene for best compatibility
    scene = trimesh.Scene(mesh)
    scene.export(out_glb)


def _filter_min_distance_grid(
    points: np.ndarray, min_dist: float, shuffle: bool = False, seed: int = 0
):
    """
    Greedy min-distance filter using a 3D spatial hash grid.
    Ensures no two returned points are within min_dist of each other.
    """
    if min_dist is None or min_dist <= 0:
        return points

    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] == 0:
        return pts

    # Optionally randomize order to avoid directional bias
    if shuffle:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(pts))
        pts_iter = pts[order]
    else:
        pts_iter = pts

    cell = float(min_dist)
    inv_cell = 1.0 / cell
    min_dist2 = cell * cell

    # grid maps (ix,iy,iz) -> list of kept points (as np arrays)
    grid = {}
    kept = []

    for p in pts_iter:
        key = tuple(np.floor(p * inv_cell).astype(np.int32))

        # Gather candidates from this cell and its neighbors
        candidates = []
        kx, ky, kz = key
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    nk = (kx + dx, ky + dy, kz + dz)
                    if nk in grid:
                        candidates.extend(grid[nk])

        # Check actual distance only against nearby candidates
        if candidates:
            c = np.asarray(candidates, dtype=np.float32)  # (M,3)
            d2 = np.sum((c - p) ** 2, axis=1)
            if np.any(d2 < min_dist2):
                continue  # too close to an existing kept point

        # Keep it
        kept.append(p)
        grid.setdefault(key, []).append(p)

    kept = np.asarray(kept, dtype=np.float32)

    # If we shuffled, order doesn't matter usually. If you want original ordering preserved,
    # don't shuffle. Otherwise you can sort kept by something if needed.
    return kept


def sample_and_export_boundary_points(
    boundary_segments,
    output_path,
    points_per_unit=100,
    min_dist=None,  # <-- hyperparameter "x" (same units as your points)
    shuffle_filter=False,  # optional: more uniform selection
    seed=0,
):
    """
    Samples points along line segments proportionally to length and exports to PLY.

    Args:
        boundary_segments: List/array of shape (N, 2, 3) or list of [p1, p2].
        output_path: File path for the .ply file.
        points_per_unit: Density of points (points per unit of length).
        min_dist: If set, enforces a minimum distance between all sampled points.
        shuffle_filter: If True, randomizes point order before filtering (often nicer).
        seed: RNG seed used when shuffle_filter=True.
    """
    all_points = []

    print(f"Sampling points with density: {points_per_unit} pts/unit...")

    for p1, p2 in boundary_segments:
        p1, p2 = np.array(p1, dtype=np.float32), np.array(p2, dtype=np.float32)

        segment_vector = p2 - p1
        segment_length = np.linalg.norm(segment_vector)

        if segment_length < 1e-8:
            continue

        num_samples = max(2, int(segment_length * points_per_unit))
        samples = np.linspace(p1, p2, num_samples, dtype=np.float32)
        all_points.append(samples)

    if not all_points:
        print("No points sampled. Check your boundary_segments input.")
        return None

    final_points = np.vstack(all_points)

    # --- Enforce minimum separation ---
    if min_dist is not None and min_dist > 0:
        before = len(final_points)
        final_points = _filter_min_distance_grid(
            final_points, min_dist, shuffle=shuffle_filter, seed=seed
        )
        after = len(final_points)
        print(f"Min-dist filtering (x={min_dist}): {before} -> {after} points")

    # Export using Trimesh PointCloud
    pc = trimesh.points.PointCloud(final_points)
    pc.visual.vertex_colors = [255, 0, 0, 255]
    pc.export(output_path)

    print(f"Successfully exported {len(final_points)} points to {output_path}")
    return final_points


# --- Main Processing ---
def process_and_export_split_files(input_path, output_path):
    print(f"Reading {input_path}...")

    # Ensure output directory exists
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    vertices = []

    # Structure: groups[group_id] = { 'faces': [], 'lines': [], 'area': 0.0, 'line_len': 0.0 }
    groups = {}
    current_group_id = None
    edge_tracker = defaultdict(set)

    re_vertex = re.compile(r"^v\s+([-\d\.eE]+)\s+([-\d\.eE]+)\s+([-\d\.eE]+)")
    re_group = re.compile(r"^g\s+(.*)")

    # 1. Parse OBJ File
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Vertices
            if line.startswith("v "):
                match = re_vertex.match(line)
                if match:
                    vertices.append(
                        [
                            float(match.group(1)),
                            float(match.group(2)),
                            float(match.group(3)),
                        ]
                    )
                continue

            # Groups (Extract Label ID)
            if line.startswith("g "):
                match = re_group.match(line)
                if match:
                    group_name = match.group(1).strip()
                    # Extract numeric ID (e.g., "face 0" -> 0)
                    numbers = re.findall(r"\d+", group_name)

                    if numbers:
                        # Convert to int to ensure "0" and "00" match if needed,
                        # but keep as string for filename if that's safer.
                        # User requested label:d, so let's treat as int.
                        group_id = int(numbers[-1])
                    else:
                        # Fallback if no number found (unlikely based on your file)
                        continue

                    if group_id not in groups:
                        groups[group_id] = {
                            "faces": [],
                            "lines": [],
                            "color": [random.randint(50, 255) for _ in range(3)]
                            + [255],
                            "total_area": 0.0,
                            "total_line_len": 0.0,
                        }
                    current_group_id = group_id
                continue

            # Faces
            if line.startswith("f ") and current_group_id is not None:
                parts = line.split()[1:]
                idxs = [int(p.split("/")[0]) - 1 for p in parts]

                # Triangulate fan
                if len(idxs) >= 3:
                    v0 = vertices[idxs[0]]
                    for i in range(1, len(idxs) - 1):
                        v1 = vertices[idxs[i]]
                        v2 = vertices[idxs[i + 1]]
                        area = triangle_area(v0, v1, v2)

                        groups[current_group_id]["faces"].append(
                            {"indices": [idxs[0], idxs[i], idxs[i + 1]], "area": area}
                        )
                        groups[current_group_id]["total_area"] += area

                        # Register edges for boundary detection
                        # Edges: (0, i), (i, i+1), (i+1, 0)
                        tri = [idxs[0], idxs[i], idxs[i + 1]]
                        edges = [
                            tuple(sorted((tri[0], tri[1]))),
                            tuple(sorted((tri[1], tri[2]))),
                            tuple(sorted((tri[2], tri[0]))),
                        ]
                        for edge in edges:
                            edge_tracker[edge].add(current_group_id)

                continue

            # Lines
            if line.startswith("l ") and current_group_id is not None:
                parts = line.split()[1:]
                idxs = []
                for p in parts:
                    try:
                        idxs.append(int(p) - 1)
                    except ValueError:
                        pass

                for i in range(len(idxs) - 1):
                    v_start = vertices[idxs[i]]
                    v_end = vertices[idxs[i + 1]]
                    length = line_length(v_start, v_end)

                    groups[current_group_id]["lines"].append(
                        {"indices": [idxs[i], idxs[i + 1]], "length": length}
                    )
                    groups[current_group_id]["total_line_len"] += length
                continue

    # --- Feature 1: Export Full Model to GLB ---
    print("Generating full model GLB...")
    scene = trimesh.Scene()

    # Convert vertices to numpy once
    np_vertices = np.array(vertices)

    for gid, data in groups.items():
        if not data["faces"]:
            continue

        group_indices = np.unique(
            np.array([f["indices"] for f in data["faces"]]).flatten()
        )
        idx_map = np.full(np.max(group_indices) + 1, -1, dtype=int)
        idx_map[group_indices] = np.arange(len(group_indices))
        new_vertices = np_vertices[group_indices]
        old_faces = np.array([f["indices"] for f in data["faces"]])
        new_faces = idx_map[old_faces]

        mesh = trimesh.Trimesh(
            vertices=new_vertices,
            faces=new_faces,
            process=False,  # Don't merge vertices or re-order
        )

        # Apply the group color
        mesh.visual.face_colors = data["color"]

        # Add to scene with a name
        scene.add_geometry(mesh, node_name=f"Group_{gid}")

    scene.export(output_path + ".glb")
    print(f"Saved {output_path}")

    # --- Feature 2: Find & Export Boundaries to GLB ---
    print("Calculating boundary edges...")
    boundary_segments = []
    boundary_v_indices = set()

    # Check our edge tracker
    for edge, connected_groups in edge_tracker.items():
        # Definition: A boundary is an edge connected to > 1 DIFFERENT groups
        if len(connected_groups) > 1:
            p1 = vertices[edge[0]]
            p2 = vertices[edge[1]]
            boundary_segments.append([p1, p2])

            # Add both vertex indices of the boundary edge
            boundary_v_indices.add(edge[0])
            boundary_v_indices.add(edge[1])

    if boundary_v_indices:
        # Convert set to sorted list for indexing
        indices = list(boundary_v_indices)
        boundary_points = np_vertices[indices]
        print(f"Found {len(boundary_points)} boundary vertices.")
        point_cloud = trimesh.points.PointCloud(vertices=boundary_points)
        point_cloud.visual.vertex_colors = [255, 0, 0, 255]
        point_cloud.export(output_path + "_boundarypoints.ply")

    # --- Feature 3 ---
    if boundary_segments:
        print(f"Found {len(boundary_segments)} boundary segments.")
        print(np.array(boundary_segments).shape)

        points = sample_and_export_boundary_points(
            boundary_segments,
            output_path + "__boundarypointsmore.ply",
            points_per_unit=200,
            min_dist=0.001,
            shuffle_filter=True,
            seed=0,
        )

        mesh = points_to_metaball_mesh(
            points,
            radius=0.0001,
            smooth_k=25,  # or try values like 10, 20, 40
            voxel_size=0.2,
            padding=0.5,
        )

        export_mesh(
            mesh,
            output_path + "_boundarysurface.ply",
            output_path + "_boundarysurface.glb",
        )


# --- Usage ---
if __name__ == "__main__":
    # Configure input here
    input_obj = "model.obj"
    output_path = "model.glb"

    if len(sys.argv) > 1:
        input_obj = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]

    process_and_export_split_files(input_obj, output_path)
