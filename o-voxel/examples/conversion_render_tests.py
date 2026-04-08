import sys

import torch
import o_voxel
import utils
import trimesh
import glob
import numpy as np
import imageio
import utils3d


import numpy as np
import torch
import trimesh


def export_lines_dual_grid_visualization(
    voxel_indices,
    dual_vertices,
    intersected,
    res,
    out_path,
    aabb_min=(-0.5, -0.5, -0.5),
    cube_rgba=(230, 230, 230, 40),
    edge_black_rgba=(0, 0, 0, 255),
    face_green_rgba=(0, 255, 0, 140),
    sphere_rgba=(255, 80, 80, 255),
    edge_radius=None,
    sphere_radius=None,
    sphere_subdivisions=1,
    edge_sections=3,
    face_convention="positive",  # "positive" or "negative"
):
    """
    Export a GLB visualization of:
      - active voxels as cubes
      - all voxel edges as black cylinders
      - intersected faces as green quads
      - dual vertices as spheres

    Parameters
    ----------
    intersected : (N, 3)
        Bool/int flags for 3 predefined faces per voxel.

    face_convention:
        "positive" means:
            flags[0] -> +x face
            flags[1] -> +y face
            flags[2] -> +z face
        "negative" means:
            flags[0] -> -x face
            flags[1] -> -y face
            flags[2] -> -z face
    """

    # ---------- Convert to CPU numpy ----------
    if torch.is_tensor(voxel_indices):
        voxel_indices = voxel_indices.detach().cpu().numpy()
    if torch.is_tensor(dual_vertices):
        dual_vertices = dual_vertices.detach().cpu().numpy()
    if torch.is_tensor(intersected):
        intersected = intersected.detach().cpu().numpy()

    voxel_indices = voxel_indices.astype(np.int64)
    dual_vertices = dual_vertices.astype(np.float32)
    intersected = intersected.astype(bool)

    aabb_min = np.asarray(aabb_min, dtype=np.float32)
    voxel_size = 1.0 / float(res)

    # Keep this only if your dual_vertices are stored in [0,1]^3.
    # If they are already in [-0.5,0.5]^3, remove this line.
    dual_vertices -= 0.5

    if edge_radius is None:
        edge_radius = 0.01 * voxel_size
    if sphere_radius is None:
        sphere_radius = 0.02 * voxel_size

    # ---------- Helpers ----------
    def canon_edge(a, b):
        a = tuple(int(x) for x in a)
        b = tuple(int(x) for x in b)
        return (a, b) if a <= b else (b, a)

    def grid_to_world(grid_pt):
        return aabb_min + np.asarray(grid_pt, dtype=np.float32) * voxel_size

    def voxel_corners(i, j, k):
        c000 = (i, j, k)
        c100 = (i + 1, j, k)
        c010 = (i, j + 1, k)
        c110 = (i + 1, j + 1, k)
        c001 = (i, j, k + 1)
        c101 = (i + 1, j, k + 1)
        c011 = (i, j + 1, k + 1)
        c111 = (i + 1, j + 1, k + 1)
        return c000, c100, c010, c110, c001, c101, c011, c111

        # All 12 cube edges (GPT)
        #       c011 ------- c111
        #     /|           /|
        # c001 ------- c101 |
        #     | |          |  |
        #     | c010 ------|-- c110
        #     |/           | /
        # c000 ------- c100

    def get_flagged_faces(i, j, k, flags, convention="positive"):
        """
        Return a list of quads (each quad = 4 corners) for flagged faces.
        """
        c000, c100, c010, c110, c001, c101, c011, c111 = voxel_corners(i, j, k)

        face_map = [
            (c000, c010, c011, c001),  # +x face
            (c000, c100, c101, c001),  # +y face
            (c000, c100, c110, c010),  # +z face
        ]

        quads = []
        for axis in range(3):
            if flags[axis]:
                quads.append(face_map[axis])
        return quads

    # ---------- 1) Cubes for active voxels ----------
    cube_template = trimesh.creation.box(extents=(voxel_size, voxel_size, voxel_size))
    cube_template.visual.face_colors = np.array(cube_rgba, dtype=np.uint8)

    cube_meshes = []
    for ijk in voxel_indices:
        center = aabb_min + (ijk.astype(np.float32) + 0.5) * voxel_size
        cube = cube_template.copy()
        cube.apply_translation(center)
        cube_meshes.append(cube)

    # ---------- 2) Build unique voxel edges ----------
    edge_keys = set()

    for ijk in voxel_indices:
        i, j, k = map(int, ijk)
        c000, c100, c010, c110, c001, c101, c011, c111 = voxel_corners(i, j, k)

        cube_edges = [
            (c000, c100),
            (c010, c110),
            (c001, c101),
            (c011, c111),  # x edges
            (c000, c001),
            (c100, c101),
            (c010, c011),
            (c110, c111),  # z edges
            (c000, c010),
            (c100, c110),
            (c001, c011),
            (c101, c111),  # y edges
        ]

        for e in cube_edges:
            edge_keys.add(canon_edge(*e))

    edge_meshes = []
    for a, b in edge_keys:
        p0 = grid_to_world(a)
        p1 = grid_to_world(b)

        cyl = trimesh.creation.cylinder(
            radius=edge_radius,
            segment=np.stack([p0, p1], axis=0),
            sections=edge_sections,
        )
        cyl.visual.face_colors = np.array(edge_black_rgba, dtype=np.uint8)
        edge_meshes.append(cyl)

    # ---------- 3) Intersected faces as green quads ----------
    face_vertices = []
    face_triangles = []

    for ijk, flags in zip(voxel_indices, intersected):
        i, j, k = map(int, ijk)
        quads = get_flagged_faces(i, j, k, flags, convention=face_convention)

        for quad in quads:
            # Convert 4 grid corners to world positions
            quad_pts = [grid_to_world(corner) for corner in quad]

            base_idx = len(face_vertices)
            face_vertices.extend(quad_pts)

            # Split quad into 2 triangles
            face_triangles.append([base_idx + 0, base_idx + 1, base_idx + 2])
            face_triangles.append([base_idx + 0, base_idx + 2, base_idx + 3])

    face_mesh = None
    if len(face_vertices) > 0:
        face_mesh = trimesh.Trimesh(
            vertices=np.asarray(face_vertices, dtype=np.float32),
            faces=np.asarray(face_triangles, dtype=np.int64),
            process=False,
        )
        face_mesh.visual.face_colors = np.array(face_green_rgba, dtype=np.uint8)

    # ---------- 4) Spheres for dual vertices ----------
    sphere_template = trimesh.creation.icosphere(
        subdivisions=sphere_subdivisions,
        radius=sphere_radius,
    )
    sphere_template.visual.face_colors = np.array(sphere_rgba, dtype=np.uint8)

    sphere_meshes = []
    for p in dual_vertices:
        sph = sphere_template.copy()
        sph.apply_translation(p)
        sphere_meshes.append(sph)

    # ---------- 5) Export ----------
    scene = trimesh.Scene()

    # # Uncomment if you want the gray cubes too
    # if cube_meshes:
    #     scene.add_geometry(trimesh.util.concatenate(cube_meshes), node_name="voxels")

    if edge_meshes:
        scene.add_geometry(trimesh.util.concatenate(edge_meshes), node_name="edges")

    if face_mesh is not None:
        scene.add_geometry(face_mesh, node_name="intersected_faces")

    if sphere_meshes:
        scene.add_geometry(
            trimesh.util.concatenate(sphere_meshes),
            node_name="dual_vertices",
        )

    scene.export(out_path)
    print(f"Saved visualization to: {out_path}")


def export_mesh_dual_grid_visualization(
    voxel_indices,
    dual_vertices,
    intersected,
    res,
    out_path,
    aabb_min=(-0.5, -0.5, -0.5),
    cube_rgba=(230, 230, 230, 40),
    edge_black_rgba=(0, 0, 0, 255),
    edge_green_rgba=(0, 255, 0, 255),
    sphere_rgba=(255, 80, 80, 255),
    edge_radius=None,
    sphere_radius=None,
    sphere_subdivisions=1,  # 2
    edge_sections=3,  # 6
):
    """
    Export a GLB visualization of:
      - active voxels as cubes
      - all voxel edges as black cylinders
      - intersected edges as green cylinders
      - dual vertices as spheres

    Parameters
    ----------
    voxel_indices : (N, 3) torch.Tensor or np.ndarray
        Active voxel integer indices.
    dual_vertices : (N, 3) torch.Tensor or np.ndarray
        Dual vertex positions in the same coordinate space as the voxelized AABB.
    intersected : (N, 3) torch.Tensor or np.ndarray
        Bool/int flags for local x/y/z edges starting at each voxel's min corner.
    res : int
        Grid resolution.
    out_path : str
        Output path, e.g. "debug_dual_grid.glb"
    aabb_min : tuple/list of 3 floats
        Minimum corner of the voxelization AABB.
    """

    # ---------- Convert to CPU numpy ----------
    if torch.is_tensor(voxel_indices):
        voxel_indices = voxel_indices.detach().cpu().numpy()
    if torch.is_tensor(dual_vertices):
        dual_vertices = dual_vertices.detach().cpu().numpy()
    if torch.is_tensor(intersected):
        intersected = intersected.detach().cpu().numpy()

    voxel_indices = voxel_indices.astype(np.int64)
    dual_vertices = dual_vertices.astype(np.float32)
    intersected = intersected.astype(bool)

    aabb_min = np.asarray(aabb_min, dtype=np.float32)
    voxel_size = 1.0 / float(res)

    dual_vertices -= 0.5

    if edge_radius is None:
        edge_radius = 0.01 * voxel_size
    if sphere_radius is None:
        sphere_radius = 0.02 * voxel_size

    # ---------- Helpers ----------
    def canon_edge(a, b):
        """Canonical undirected edge key in integer grid coordinates."""
        a = tuple(int(x) for x in a)
        b = tuple(int(x) for x in b)
        return (a, b) if a <= b else (b, a)

    def grid_to_world(grid_pt):
        """Convert integer grid corner coords to object/world coords."""
        return aabb_min + np.asarray(grid_pt, dtype=np.float32) * voxel_size

    # ---------- 1) Cubes for active voxels ----------
    cube_template = trimesh.creation.box(extents=(voxel_size, voxel_size, voxel_size))
    cube_template.visual.face_colors = np.array(cube_rgba, dtype=np.uint8)

    cube_meshes = []
    for ijk in voxel_indices:
        center = aabb_min + (ijk.astype(np.float32) + 0.5) * voxel_size
        cube = cube_template.copy()
        cube.apply_translation(center)
        cube_meshes.append(cube)

    # ---------- 2) Build unique voxel edges ----------
    # We deduplicate edges globally so shared edges between adjacent voxels
    # are exported only once.
    #
    # Edge color rule:
    #   default = black
    #   if an edge is marked intersected by any voxel, color it green
    edge_is_green = {}

    for ijk, flags in zip(voxel_indices, intersected):
        i, j, k = map(int, ijk)

        c000 = (i, j, k)
        c100 = (i + 1, j, k)
        c010 = (i, j + 1, k)
        c110 = (i + 1, j + 1, k)
        c001 = (i, j, k + 1)
        c101 = (i + 1, j, k + 1)
        c011 = (i, j + 1, k + 1)
        c111 = (i + 1, j + 1, k + 1)

        # All 12 cube edges (GPT)
        #       c011 ------- c111
        #     /|           /|
        # c001 ------- c101 |
        #     | |          |  |
        #     | c010 ------|-- c110
        #     |/           | /
        # c000 ------- c100

        cube_edges = [
            (c000, c100),
            (c010, c110),
            (c001, c101),
            (c011, c111),  # x edges
            (c000, c001),
            (c100, c101),
            (c010, c011),
            (c110, c111),  # z edges
            (c000, c010),
            (c100, c110),
            (c001, c011),
            (c101, c111),  # y edges
        ]

        # # All 12 cube edges (From flexible_dual_grid.py)
        # # [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],     # x-axis
        # # [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],     # y-axis
        # # [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],     # z-axis

        for e in cube_edges:
            key = canon_edge(*e)
            edge_is_green.setdefault(key, False)

        # Local intersected edges from voxel min corner
        if flags[0]:
            edge_is_green[canon_edge(c011, c111)] = True
        if flags[1]:
            edge_is_green[canon_edge(c101, c111)] = True
        if flags[2]:
            edge_is_green[canon_edge(c110, c111)] = True

    edge_meshes = []
    for (a, b), is_green in edge_is_green.items():
        p0 = grid_to_world(a)
        p1 = grid_to_world(b)

        cyl = trimesh.creation.cylinder(
            radius=edge_radius,
            segment=np.stack([p0, p1], axis=0),
            sections=edge_sections,
        )
        cyl.visual.face_colors = np.array(
            edge_green_rgba if is_green else edge_black_rgba,
            dtype=np.uint8,
        )
        edge_meshes.append(cyl)

    # ---------- 3) Spheres for dual vertices ----------
    sphere_template = trimesh.creation.icosphere(
        subdivisions=sphere_subdivisions,
        radius=sphere_radius,
    )
    sphere_template.visual.face_colors = np.array(sphere_rgba, dtype=np.uint8)

    sphere_meshes = []
    for p in dual_vertices:
        sph = sphere_template.copy()
        sph.apply_translation(p)
        sphere_meshes.append(sph)

    # ---------- 4) Export ----------
    scene = trimesh.Scene()

    # if cube_meshes:
    # scene.add_geometry(trimesh.util.concatenate(cube_meshes), node_name="voxels")
    if edge_meshes:
        scene.add_geometry(trimesh.util.concatenate(edge_meshes), node_name="edges")
    if sphere_meshes:
        scene.add_geometry(
            trimesh.util.concatenate(sphere_meshes), node_name="dual_vertices"
        )

    scene.export(out_path)
    print(f"Saved visualization to: {out_path}")


def render(position, base_color, glb_path, type, voxel_size=1.0 / 64):
    # Setup camera
    extr = utils3d.extrinsics_look_at(
        eye=torch.tensor([1.2, 0.5, 1.2]),
        look_at=torch.tensor([0.0, 0.0, 0.0]),
        up=torch.tensor([0.0, 1.0, 0.0]),
    ).cuda()
    intr = utils3d.intrinsics_from_fov_xy(
        fov_x=torch.deg2rad(torch.tensor(45.0)),
        fov_y=torch.deg2rad(torch.tensor(45.0)),
    ).cuda()

    # Render
    renderer = o_voxel.rasterize.VoxelRenderer(
        rendering_options={"resolution": 1024, "ssaa": 2}
    )
    output = renderer.render(
        position=position,  # Voxel centers
        attrs=base_color,  # Color/Opacity etc.
        voxel_size=voxel_size,
        extrinsics=extr,
        intrinsics=intr,
    )
    image = np.clip(output.attr.permute(1, 2, 0).cpu().numpy() * 255, 0, 255).astype(
        np.uint8
    )
    imageio.imwrite(glb_path.replace(".glb", f"_{type}.png").split("/")[-1], image)


def process_lines(glb_path):
    def load_obj_lines_as_path(obj_file):
        vertices = []
        segments = []
        with open(obj_file, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                tag = parts[0]

                if tag == "v":
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif tag == "l":
                    # OBJ indices are 1-based, and may appear like "12" or "12/5"
                    idx = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                    # turn polyline into pairwise segments
                    for a, b in zip(idx[:-1], idx[1:]):
                        segments.append([vertices[a], vertices[b]])

        segments = np.asarray(segments, dtype=float)
        path = trimesh.load_path(segments)
        return path

    print(f"\n\nProcessing {glb_path}...\n\n")
    ext = glb_path.split(".")[-1].lower()
    # asset = trimesh.load(glb_path, process=False)
    asset = load_obj_lines_as_path(glb_path)

    print(type(asset))  # trimesh.path.path.Path3D
    print(asset.vertices.shape)  # (n, 3)

    # 0. Normalize asset to unit cube
    bounds = asset.bounds  # shape (2, 3): [min, max]
    center = bounds.mean(axis=0)
    extent = (bounds[1] - bounds[0]).max()
    scale = 0.99999 / extent  # avoid edge-touching numerical issues

    # Option A: if these methods exist in your trimesh version
    asset.apply_translation(-center)
    asset.apply_scale(scale)

    lines = []

    for e in asset.entities:
        vertices = asset.vertices[e.points]
        for i in range(len(vertices) - 1):
            lines.append((i, i + 1))
    lines = np.array(lines)

    print(lines.shape)  # (num_lines, 2)

    # 1. Geometry Voxelization (Flexible Dual Grid)
    # Returns: occupied indices, dual vertices (QEF solution), and edge intersected
    vertices = torch.from_numpy(vertices).float()
    lines = torch.from_numpy(lines).long()
    voxel_indices, dual_vertices, intersected = (
        o_voxel.convert.lines_to_flexible_dual_grid(
            vertices,
            lines,
            grid_size=RES,  # Resolution
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],  # Axis-aligned bounding box
            regularization_weight=1e-2,  # Regularization term weight in QEF
            timing=True,
        )
    )

    ## sort to ensure align between geometry and material voxelization
    vid = o_voxel.serialize.encode_seq(voxel_indices)
    mapping = torch.argsort(vid)
    voxel_indices = voxel_indices[mapping]
    dual_vertices = dual_vertices[mapping]
    intersected = intersected[mapping]

    ## packing
    dual_vertices_quant = dual_vertices * RES - voxel_indices
    dual_vertices_quant = (torch.clamp(dual_vertices_quant, 0, 1) * 255).type(
        torch.uint8
    )
    intersected_quant = (
        intersected[:, 0:1] + 2 * intersected[:, 1:2] + 4 * intersected[:, 2:3]
    ).type(torch.uint8)

    print(f"voxel_indices: {voxel_indices.shape}, {voxel_indices.dtype}")
    print(f"dual_vertices: {dual_vertices.shape}, {dual_vertices.dtype}")
    print(f"intersected: {intersected.shape}, {intersected.dtype}")

    # ------------------------------------------------------------
    # ) Export dual vertices and intersected faces
    # ------------------------------------------------------------

    export_lines_dual_grid_visualization(
        voxel_indices=voxel_indices,
        dual_vertices=dual_vertices,
        intersected=intersected,
        res=RES,
        out_path=glb_path.replace("." + ext, "_dual_grid_debug." + ext).split("/")[-1],
    )

    # ------------------------------------------------------------
    # ) Export input vertices and lines to .obj
    # ------------------------------------------------------------
    with open(
        glb_path.replace("." + ext, "_original_lines.obj").split("/")[-1], "w"
    ) as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for line in lines:
            # OBJ format is 1-indexed
            f.write(f"l {line[0]+1} {line[1]+1}\n")


def process_mesh(glb_path):
    print(f"\n\nProcessing {glb_path}...\n\n")
    asset = trimesh.load(glb_path)

    # 0. Normalize asset to unit cube
    aabb = asset.bounding_box.bounds
    center = (aabb[0] + aabb[1]) / 2
    scale = 0.99999 / (aabb[1] - aabb[0]).max()  # To avoid numerical issues
    asset.apply_translation(-center)
    asset.apply_scale(scale)

    # 1. Geometry Voxelization (Flexible Dual Grid)
    # Returns: occupied indices, dual vertices (QEF solution), and edge intersected
    mesh = asset.to_mesh()
    print(
        f"Mesh has {len(mesh.vertices)} vertices and {len(mesh.faces)} faces after normalization."
    )

    vertices = torch.from_numpy(mesh.vertices).float()
    faces = torch.from_numpy(mesh.faces).long()
    voxel_indices, dual_vertices, intersected = (
        o_voxel.convert.mesh_to_flexible_dual_grid(
            vertices,
            faces,
            grid_size=RES,  # Resolution
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],  # Axis-aligned bounding box
            face_weight=1.0,  # Face term weight in QEF
            boundary_weight=0.2,  # Boundary term weight in QEF
            regularization_weight=1e-2,  # Regularization term weight in QEF
            timing=True,
        )
    )
    ## sort to ensure align between geometry and material voxelization
    vid = o_voxel.serialize.encode_seq(voxel_indices)
    mapping = torch.argsort(vid)
    voxel_indices = voxel_indices[mapping]
    dual_vertices = dual_vertices[mapping]
    intersected = intersected[mapping]

    ## packing
    dual_vertices_quant = dual_vertices * RES - voxel_indices
    dual_vertices_quant = (torch.clamp(dual_vertices_quant, 0, 1) * 255).type(
        torch.uint8
    )
    intersected_quant = (
        intersected[:, 0:1] + 2 * intersected[:, 1:2] + 4 * intersected[:, 2:3]
    ).type(torch.uint8)

    print(f"voxel_indices: {voxel_indices.shape}, {voxel_indices.dtype}")
    print(f"dual_vertices: {dual_vertices.shape}, {dual_vertices.dtype}")
    print(f"intersected: {intersected.shape}, {intersected.dtype}")

    # o_voxel.io.write(glb_path.replace(".glb", ".vxz"), voxel_indices, attributes)

    # render(
    #     (voxel_indices / RES - 0.5).cuda(),
    #     (voxel_indices / RES).cuda(),
    #     glb_path,
    #     type="voxels",
    #     voxel_size=1.0 / RES,
    # )

    # render(
    #     (dual_vertices).cuda() * 0.7,
    #     torch.clamp(dual_vertices + 0.5, 0.0, 1.0).cuda(),
    #     glb_path,
    #     type="dualvertices",
    #     voxel_size=0.25 / RES,
    # )

    # ------------------------------------------------------------
    # ) Export dual vertices and intersected edges
    # ------------------------------------------------------------

    export_mesh_dual_grid_visualization(
        voxel_indices=voxel_indices,
        dual_vertices=dual_vertices,
        intersected=intersected,
        res=RES,
        out_path=glb_path.replace(".glb", f"_{RES}_dual_grid_debug.glb").split("/")[-1],
    )

    mesh.export(glb_path.replace(".glb", "_original_mesh.obj").split("/")[-1])


if __name__ == "__main__":
    # Read RES from stdin argumentls
    RES = sys.argv[1] if len(sys.argv) > 1 else 16
    RES = int(RES)

    # glb_paths = glob.glob(
    #     "/home/vthamizharas/Documents/TRELLIS.2/datasets/ObjaverseXL_sketchfab/raw/hf-objaverse-v1/glbs/**/*.glb"
    # )
    # glb_paths = glob.glob("shapes/pcb_vise_segment1_boundarysurface.glb")

    glb_paths = glob.glob(
        "/home/vthamizharas/Documents/TRELLIS.2/brep_parquet_outputs/r=e100_k=8e2_v=3e3/*.glb"
    )

    for glb_path in glb_paths:
        ext = glb_path.split(".")[-1].lower()
        try:
            if ext == "obj":
                process_lines(glb_path)
            elif ext == "glb":
                process_mesh(glb_path)
        except Exception as e:
            print(f"Error processing {glb_path}: {e}")
