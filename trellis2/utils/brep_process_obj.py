import os
import sys
import random
import math
import re
import numpy as np
import trimesh
from collections import defaultdict
from itertools import combinations

from trellis2.utils.brep_helpers import *
from trellis2.datasets.abc_data import *
from trellis2.utils.surface_gen import *


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
def process_obj_to_trellismesh(input_path, output_path, useMetaballs=True):

    print(f"Reading {input_path}...")

    vertices = []

    # groups[group_id] = { 'faces': [], 'lines': [], 'color': [...], ... }
    groups = {}
    current_group_id = None

    # Track edges by GEOMETRY, not by raw vertex indices
    # edge_tracker[geom_edge] = {
    #     "groups": set(...),
    #     "points": (p1, p2)
    # }
    edge_tracker = defaultdict(lambda: {"groups": set(), "points": None})
    group_name2id = {}
    group_id_counter = 0

    re_vertex = re.compile(r"^v\s+([-+\d\.eE]+)\s+([-+\d\.eE]+)\s+([-+\d\.eE]+)")
    re_group = re.compile(r"^g\s+(.*)")

    def quantize_point(v, ndigits=6):
        # Rounds coordinates so duplicated seam vertices across groups match
        return tuple(round(float(c), ndigits) for c in v)

    def make_geom_edge(idx_a, idx_b):
        p1 = vertices[idx_a]
        p2 = vertices[idx_b]
        q1 = quantize_point(p1)
        q2 = quantize_point(p2)
        # Sort endpoints so edge direction doesn't matter
        geom_edge = tuple(sorted((q1, q2)))
        return geom_edge, p1, p2

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

            # Groups
            if line.startswith("g "):
                match = re_group.match(line)
                if match:
                    group_name = match.group(1).strip()
                    numbers = re.findall(r"\d+", group_name)
                    if group_name not in group_name2id.keys():
                        group_name2id[group_name] = group_id_counter
                        group_id_counter += 1
                    # if not numbers:
                    #     current_group_id = None
                    #     continue
                    # group_id = int(numbers[-1])
                    group_id = group_name2id[group_name]

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

                # Triangulate polygon as fan
                if len(idxs) >= 3:
                    v0 = vertices[idxs[0]]
                    for i in range(1, len(idxs) - 1):
                        tri = [idxs[0], idxs[i], idxs[i + 1]]

                        v1 = vertices[tri[1]]
                        v2 = vertices[tri[2]]
                        area = triangle_area(v0, v1, v2)

                        groups[current_group_id]["faces"].append(
                            {"indices": tri, "area": area}
                        )
                        groups[current_group_id]["total_area"] += area

                        # Register triangle edges by GEOMETRIC position
                        tri_edges = [
                            (tri[0], tri[1]),
                            (tri[1], tri[2]),
                            (tri[2], tri[0]),
                        ]

                        for a, b in tri_edges:
                            geom_edge, p1, p2 = make_geom_edge(a, b)
                            edge_tracker[geom_edge]["groups"].add(current_group_id)

                            if edge_tracker[geom_edge]["points"] is None:
                                edge_tracker[geom_edge]["points"] = (p1, p2)

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

    # --- Feature 1: Find & Export Boundaries to GLB ---

    print("Generating full model GLB...")
    scene = trimesh.Scene()
    np_vertices = np.array(vertices)

    print("Calculating boundary edges...")
    boundary_segments = []
    boundary_records = []

    for edge, info in edge_tracker.items():
        connected_groups = sorted(info["groups"])

        # Boundary between DIFFERENT groups
        if len(connected_groups) > 1:
            p1, p2 = info["points"]
            boundary_segments.append([p1, p2])

            for ga, gb in combinations(connected_groups, 2):
                boundary_records.append(
                    {
                        "groups": [int(ga), int(gb)],
                        "points": [p1, p2],
                    }
                )

    print(f"Found {len(boundary_segments)} boundary segments.")
    # Optional: export your detected B-Rep edges as OBJ lines.
    export_segments_obj(boundary_segments, output_path + "_boundary.obj")

    if len(boundary_segments) == 0 or len(boundary_segments) > 80:
        return False

    # --- Feature 2: Export Full Model to GLB ---

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
            process=False,
        )
        mesh.visual.face_colors = data["color"]
        scene.add_geometry(mesh, node_name=f"Group_{gid}")

    scene.export(output_path + ".glb")
    print(f"Saved {output_path}.glb")

    # --- Feature 3 ---
    isTubular = False

    if isTubular:
        boundary_segments, centroid, furthest_distance = normalize_point_cloud_list(
            boundary_segments
        )
        points, points_segments = sample_and_export_boundary_points_new(
            boundary_segments,
            points_per_unit=4e2,
            min_dist=1e-8,
            shuffle_filter=True,
            seed=0,
        )

        if useMetaballs:
            # Normalize points
            # points = normalize_point_cloud(points) * 1.5
            scale_factor = 4
            mesh = points_to_metaball_mesh_new(
                points * scale_factor,
                radius=1e-13,  # 1e-100,
                smooth_k=9e1,  # 1e3,
                voxel_size=5e-3,  # 2.5e-3,
                padding=4e-1,
                eps=1e-6,
            )

        else:
            # points = normalize_point_cloud(points) * 1.2
            scale_factor = 1.3
            mesh = points_to_convolution_surface_mesh(
                points * scale_factor,
                radius=4e-3,
                support_radius=8e-3,  # smaller = less cross-branch blending
                voxel_size=5e-3,
                padding=4e-1,
                segments=points_segments * scale_factor,  # boundary_v_indices
            )

        output_path = f"{output_path}_{'metaball' if useMetaballs else 'convexsurf'}"

        # Export Points using Trimesh PointCloud
        points = unnormalize_point_cloud(points, centroid, furthest_distance)
        pc = trimesh.points.PointCloud(points)
        pc.visual.vertex_colors = [255, 0, 0, 255]
        pc.export(output_path + "_boundarypointsmore.ply")

        # Export Mesh
        mesh.vertices = unnormalize_point_cloud(
            mesh.vertices / scale_factor, centroid, furthest_distance
        )
        export_mesh(
            mesh,
            output_path + "_boundarysurface.ply",
            output_path + "_boundarysurface.glb",
        )
    else:

        # # Convert group dictionary into flat mesh arrays plus one B-Rep patch id per triangle.
        # V, F, face_patch_ids = arrays_from_importer_groups(vertices, groups)

        # # Optional: infer topological patch adjacency from shared/quantized geometric edges.
        # adjacent_patch_pairs, inferred_boundary_segments = patch_adjacency_from_mesh(
        #     V,
        #     F,
        #     face_patch_ids,
        #     ndigits=6,
        # )

        # Full GVD: includes sheets between any two patches that become nearest neighbors
        # inside the sampled bounding box.
        # gvd = compute_generalized_voronoi(
        #     V,
        #     F,
        #     face_patch_ids,
        #     resolution=160,
        #     padding=0.08,
        #     max_distance=None,
        #     inside_only=False,
        #     allowed_pairs=None,
        #     neighborhood=26,
        #     batch_size=200_000,
        #     verbose=True,
        # )
        # gvd = compute_generalized_voronoi(
        #     V,
        #     F,
        #     face_patch_ids,
        #     resolution=160,
        #     padding=0.08,
        #     allowed_pairs=adjacent_patch_pairs,
        #     neighborhood=26,
        #     verbose=True,
        #     inside_only=True,
        # )

        # paths = gvd.export(output_path + "_boundary")

        planes, sheets = build_planar_gvd_sheets(
            vertices,
            groups,
            boundary_records=boundary_records,
            mode="bbox",
            branches="both",
            padding=0.08,
        )
        paths = export_planar_gvd(
            planes,
            sheets,
            output_path + "_boundary",
        )
        print(paths)

        # Optional: export your detected B-Rep edges as OBJ lines.
        # export_segments_obj(boundary_segments, output_path + "_boundary.obj")

    return True


# --- Usage ---
if __name__ == "__main__":
    # Configure input here
    input_obj = None
    output_path = None
    root = Path("datasets/Fusion360/")

    if len(sys.argv) > 1:
        input_obj = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]

    if input_obj is not None and output_path is not None:
        process_obj_to_trellismesh(input_obj, output_path)
    else:
        import glob

        fusion360_objs = list(root.glob("raw/*/*.obj"))

        fusion360_objs = sort_objs(
            fusion360_objs, root / "raw_processed/obj_complexity_cache.json"
        )

        processed_count = 0
        for i, obj in enumerate(fusion360_objs):
            obj_path = obj["file"]
            try:
                base_name = os.path.splitext(os.path.basename(obj_path))[0]
                if base_name != "assembly":
                    print(f"Processing {obj_path}...")

                    useMetaballs = True
                    output_base = os.path.join(
                        root / "raw_processed/",
                        base_name,
                    )
                    # success = process_obj_to_trellismesh(
                    #     obj_path, output_base, useMetaballs
                    # )
                    success, _, _ = export_patch_aligned_ribbons_for_trellis2(
                        input_obj_path=obj_path,
                        output_prefix=output_base,
                        width_fraction=0.002,
                        normal_offset_fraction_of_width=0.25,
                        include_original_surface=True,
                        include_inter_patch_edges=True,
                        include_obj_lines=True,
                        # include_single_patch_mesh_boundaries=False,
                    )
                    if success:
                        processed_count += 1

                if processed_count >= 100:
                    break
            except Exception as e:
                print(f"Error processing {obj_path}: {e}")
