import sys
import random
import math
import re
import numpy as np
import trimesh
from collections import defaultdict


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


# --- Main Processing ---
def process_obj_to_trellismesh(input_path, output_path):
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
            radius=0.00001,
            smooth_k=50,  # or try values like 10, 20, 40
            voxel_size=0.04,
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

    process_obj_to_trellismesh(input_obj, output_path)
