import re
import random
import math
import numpy as np
import glob
from collections import defaultdict
import sys

import utils
import trimesh
import imageio
import utils3d

import torch
import o_voxel


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

        # Local intersected edges from voxel min corner
        # edge_is_green[canon_edge(c011, c111)] = True
        # edge_is_green[canon_edge(c101, c111)] = True
        # edge_is_green[canon_edge(c110, c111)] = True

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


def import_obj_with_groups(input_path):
    print(f"Reading {input_path}...")

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

    return vertices, groups, edge_tracker


def process_lines(vertices, lines, filename1, filename2):

    print(f"\nProcessing lines {filename1} for dual grid conversion...\n")

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

    print(f"voxel_indices: {voxel_indices.shape}, {voxel_indices.dtype}")
    print(f"dual_vertices: {dual_vertices.shape}, {dual_vertices.dtype}")
    print(f"intersected: {intersected.shape}, {intersected.dtype}")

    # ------------------------------------------------------------
    # Export dual vertices and intersected faces
    # ------------------------------------------------------------

    export_lines_dual_grid_visualization(
        voxel_indices=voxel_indices,
        dual_vertices=dual_vertices,
        intersected=intersected,
        res=RES,
        out_path=filename1,
    )

    # ------------------------------------------------------------
    # Export input vertices and lines to .obj
    # ------------------------------------------------------------
    with open(filename2, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for line in lines:
            # OBJ format is 1-indexed
            f.write(f"l {line[0]+1} {line[1]+1}\n")


def process_mesh(mesh, filename):
    """
    Process a triangular mesh and convert it to a flexible dual grid.
    Input:
    - mesh: a trimesh.Trimesh object containing the geometry to convert.
    - filename: output filename for the visualization GLB.
    """
    print(f"\nProcessing mesh {filename} for dual grid conversion...\n")

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

    print(f"voxel_indices: {voxel_indices.shape}, {voxel_indices.dtype}")
    print(f"dual_vertices: {dual_vertices.shape}, {dual_vertices.dtype}")
    print(f"intersected: {intersected.shape}, {intersected.dtype}")

    # ------------------------------------------------------------
    # ) Export dual vertices and intersected edges
    # ------------------------------------------------------------

    export_mesh_dual_grid_visualization(
        voxel_indices=voxel_indices,
        dual_vertices=dual_vertices,
        intersected=intersected,
        res=RES,
        out_path=filename,
    )


if __name__ == "__main__":
    # Read RES from stdin argument
    RES = sys.argv[1] if len(sys.argv) > 1 else 16
    RES = int(RES)

    print(f"Using resolution: {RES}")

    # shape_paths = glob.glob(
    #     "/home/vthamizharas/Documents/TRELLIS.2/assets/breps/abc/00000012_f1*"
    #     # "/home/vthamizharas/Documents/TRELLIS.2/datasets/ObjaverseXL_sketchfab/raw/hf-objaverse-v1/glbs/**/*.glb"
    # )
    shape_paths = glob.glob("shapes/pcb_vise_segment1.obj")

    for shape_path in shape_paths:
        ext = shape_path.split(".")[-1].lower()
        basename = shape_path.split("/")[-1].split(".")[0]
        basename = f"{basename}_res{RES}"

        vertices, groups, edge_tracker = import_obj_with_groups(shape_path)

        # 0. Normalize vertices to unit cube
        aabb = [
            [min(v[i] for v in vertices), max(v[i] for v in vertices)] for i in range(3)
        ]
        center = [(aabb[i][0] + aabb[i][1]) / 2 for i in range(3)]
        scale = 0.99999 / max(aabb[i][1] - aabb[i][0] for i in range(3))
        vertices = [[(v[i] - center[i]) * scale for i in range(3)] for v in vertices]

        # Convert vertices to numpy once
        np_vertices = np.array(vertices)

        # ---------------------------------------------------------------------
        # Compute Dual Grid for triangular mesh faces and export visualization
        # ---------------------------------------------------------------------
        scene = trimesh.Scene()
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

            break

        process_mesh(scene.to_mesh(), f"outputs/{basename}_mesh_dual_grid.glb")

        scene.export(f"outputs/{basename}_original_mesh.glb")

        # -------------------------------------------------------------
        # Compute Dual Grid for line segments and export visualization
        # -------------------------------------------------------------
        print("Calculating boundary edges...")
        boundary_edges = []
        # Check our edge tracker
        print(f"Total unique edges: {len(edge_tracker)}")
        for edge, connected_groups in edge_tracker.items():
            # Definition: A boundary is an edge connected to > 1 DIFFERENT groups
            if len(connected_groups) > 1 and connected_groups == {0, 1}:
                boundary_edges.append([edge[0], edge[1]])

        old_to_new = {}
        boundary_vertices_list = []
        boundary_lines = []

        for old_i, old_j in boundary_edges:
            if old_i not in old_to_new:
                old_to_new[old_i] = len(boundary_vertices_list)
                boundary_vertices_list.append(np_vertices[old_i])

            if old_j not in old_to_new:
                old_to_new[old_j] = len(boundary_vertices_list)
                boundary_vertices_list.append(np_vertices[old_j])

            new_i = old_to_new[old_i]
            new_j = old_to_new[old_j]
            boundary_lines.append([new_i, new_j])

        boundary_vertices = np.asarray(boundary_vertices_list, dtype=np_vertices.dtype)

        process_lines(
            boundary_vertices,
            np.array(boundary_lines),
            f"outputs/{basename}_line_dual_grid.glb",
            f"outputs/{basename}_original_lines.obj",
        )
