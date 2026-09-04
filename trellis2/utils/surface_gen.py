import json
import random
import re
from collections import defaultdict

import numpy as np
import trimesh

EPS = 1e-12


def normalize(v, fallback=None):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < EPS:
        if fallback is None:
            return None
        return np.asarray(fallback, dtype=float)
    return v / n


def triangle_area_and_normal(a, b, c):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    n = np.cross(b - a, c - a)
    length = np.linalg.norm(n)
    if length < EPS:
        return 0.0, np.array([0.0, 0.0, 1.0])

    return 0.5 * length, n / length


def line_length(a, b):
    return float(
        np.linalg.norm(np.asarray(b, dtype=float) - np.asarray(a, dtype=float))
    )


def parse_obj_index(token, num_vertices):
    """
    Supports OBJ indices like:
        f 1 2 3
        f 1/2/3 4/5/6 7/8/9
        f -3 -2 -1
    """
    raw = int(token.split("/")[0])
    if raw < 0:
        return num_vertices + raw
    return raw - 1


def parse_obj_with_patch_info(input_path, quantize_ndigits=6):
    vertices = []

    groups = {}
    group_name2id = {}
    group_id2name = {}
    group_id_counter = 0
    current_group_id = None

    # Mesh edges from triangle faces.
    edge_tracker = defaultdict(
        lambda: {
            "groups": set(),
            "points": None,
            "face_normals_by_group": defaultdict(list),  # gid -> [(normal, area), ...]
        }
    )

    # Explicit OBJ `l` segments.
    line_tracker = defaultdict(
        lambda: {
            "groups": set(),
            "points": None,
        }
    )

    re_vertex = re.compile(r"^v\s+([-+\d\.eE]+)\s+([-+\d\.eE]+)\s+([-+\d\.eE]+)")
    re_group = re.compile(r"^g\s+(.*)")

    def quantize_point(v):
        return tuple(round(float(c), quantize_ndigits) for c in v)

    def make_geom_edge_from_points(p0, p1):
        q0 = quantize_point(p0)
        q1 = quantize_point(p1)
        return tuple(sorted((q0, q1)))

    def make_geom_edge_from_indices(i0, i1):
        p0 = vertices[i0]
        p1 = vertices[i1]
        return make_geom_edge_from_points(p0, p1), p0, p1

    def get_or_create_group(group_name):
        nonlocal group_id_counter

        if group_name not in group_name2id:
            gid = group_id_counter
            group_id_counter += 1

            group_name2id[group_name] = gid
            group_id2name[gid] = group_name

            groups[gid] = {
                "name": group_name,
                "faces": [],
                "lines": [],
                "color": [random.randint(50, 255) for _ in range(3)] + [255],
                "total_area": 0.0,
                "total_line_len": 0.0,
            }

        return group_name2id[group_name]

    with open(input_path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

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

            if line.startswith("g "):
                match = re_group.match(line)
                if match:
                    group_name = match.group(1).strip()
                    current_group_id = get_or_create_group(group_name)
                continue

            if line.startswith("f ") and current_group_id is not None:
                parts = line.split()[1:]
                idxs = [parse_obj_index(p, len(vertices)) for p in parts]

                # Fan triangulation.
                if len(idxs) >= 3:
                    for i in range(1, len(idxs) - 1):
                        tri = [idxs[0], idxs[i], idxs[i + 1]]

                        p0 = vertices[tri[0]]
                        p1 = vertices[tri[1]]
                        p2 = vertices[tri[2]]

                        area, normal = triangle_area_and_normal(p0, p1, p2)

                        groups[current_group_id]["faces"].append(
                            {
                                "indices": tri,
                                "area": area,
                                "normal": normal.tolist(),
                            }
                        )
                        groups[current_group_id]["total_area"] += area

                        tri_edges = [
                            (tri[0], tri[1]),
                            (tri[1], tri[2]),
                            (tri[2], tri[0]),
                        ]

                        for a, b in tri_edges:
                            edge_key, ep0, ep1 = make_geom_edge_from_indices(a, b)

                            info = edge_tracker[edge_key]
                            info["groups"].add(current_group_id)

                            if info["points"] is None:
                                info["points"] = (ep0, ep1)

                            info["face_normals_by_group"][current_group_id].append(
                                (normal, area)
                            )

                continue

            if line.startswith("l ") and current_group_id is not None:
                parts = line.split()[1:]
                idxs = []

                for p in parts:
                    try:
                        idxs.append(parse_obj_index(p, len(vertices)))
                    except ValueError:
                        pass

                for i in range(len(idxs) - 1):
                    i0 = idxs[i]
                    i1 = idxs[i + 1]

                    p0 = vertices[i0]
                    p1 = vertices[i1]
                    length = line_length(p0, p1)

                    groups[current_group_id]["lines"].append(
                        {"indices": [i0, i1], "length": length}
                    )
                    groups[current_group_id]["total_line_len"] += length

                    edge_key = make_geom_edge_from_points(p0, p1)
                    info = line_tracker[edge_key]
                    info["groups"].add(current_group_id)

                    if info["points"] is None:
                        info["points"] = (p0, p1)

                continue

    return {
        "vertices": np.asarray(vertices, dtype=float),
        "groups": groups,
        "group_name2id": group_name2id,
        "group_id2name": group_id2name,
        "edge_tracker": edge_tracker,
        "line_tracker": line_tracker,
    }


def compute_group_normals(parsed):
    group_normals = {}

    for gid, group in parsed["groups"].items():
        accum = np.zeros(3, dtype=float)

        for face in group["faces"]:
            n = np.asarray(face["normal"], dtype=float)
            area = float(face["area"])
            accum += area * n

        group_normals[gid] = normalize(accum, fallback=np.array([0.0, 0.0, 1.0]))

    return group_normals


def fallback_patch_normal_from_segment(p0, p1):
    """
    Used only when a patch normal is unavailable.
    This is not truly patch-aligned, but it keeps the code robust.
    """
    t = normalize(np.asarray(p1) - np.asarray(p0), fallback=np.array([1.0, 0.0, 0.0]))

    candidate_axes = [
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    ]

    axis = min(candidate_axes, key=lambda a: abs(np.dot(a, t)))
    n = normalize(np.cross(t, axis), fallback=np.array([0.0, 0.0, 1.0]))
    return n


def get_local_patch_normal(edge_key, gid, parsed, group_normals):
    """
    Prefer local triangle normals incident on this boundary edge.
    Fall back to the patch/group average normal.
    """
    edge_info = parsed["edge_tracker"].get(edge_key)

    if edge_info is not None:
        local = edge_info["face_normals_by_group"].get(gid, [])
        if local:
            accum = np.zeros(3, dtype=float)
            for normal, area in local:
                accum += float(area) * np.asarray(normal, dtype=float)

            n = normalize(accum)
            if n is not None:
                return n

    if gid in group_normals:
        return group_normals[gid]

    return None


def make_patch_aligned_ribbon(
    p0,
    p1,
    patch_normal,
    width,
    normal_offset=0.0,
):
    """
    Builds one rectangular ribbon, triangulated as two triangles.

    The ribbon lies in the tangent plane defined by:
        segment tangent t
        ribbon direction r = patch_normal x t

    p0 and p1 are the original segment endpoints.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)

    t = normalize(p1 - p0)
    if t is None:
        return None

    n = normalize(patch_normal)
    if n is None:
        n = fallback_patch_normal_from_segment(p0, p1)

    # Remove any component of n along the edge tangent.
    # This avoids numerical issues if the normal is not perfectly perpendicular.
    n = n - np.dot(n, t) * t
    n = normalize(n)

    if n is None:
        n = fallback_patch_normal_from_segment(p0, p1)

    # Tangent-plane direction perpendicular to the segment.
    r = normalize(np.cross(n, t))
    if r is None:
        return None

    half_width = 0.5 * width

    c0 = p0 + normal_offset * n
    c1 = p1 + normal_offset * n

    v0 = c0 + half_width * r
    v1 = c0 - half_width * r
    v2 = c1 + half_width * r
    v3 = c1 - half_width * r

    # Oriented to approximately match patch_normal.
    verts = np.asarray([v0, v1, v2, v3], dtype=float)
    faces = np.asarray(
        [
            [0, 1, 2],
            [1, 3, 2],
        ],
        dtype=np.int64,
    )

    return {
        "vertices": verts,
        "faces": faces,
        "patch_normal": n,
        "ribbon_direction": r,
        "centerline_start": p0,
        "centerline_end": p1,
    }


def collect_feature_segments(
    parsed,
    include_inter_patch_edges=True,
    include_obj_lines=True,
    include_single_patch_mesh_boundaries=False,
):
    """
    Returns geometric edge keys that should receive ribbons.

    include_inter_patch_edges:
        Uses triangle mesh edges shared by more than one OBJ group.

    include_obj_lines:
        Uses explicit OBJ `l` segments.

    include_single_patch_mesh_boundaries:
        Also includes open mesh edges belonging to only one patch.
        Usually useful for outer boundaries.
    """
    candidates = {}

    if include_inter_patch_edges or include_single_patch_mesh_boundaries:
        for edge_key, info in parsed["edge_tracker"].items():
            num_groups = len(info["groups"])

            use_edge = False

            if include_inter_patch_edges and num_groups > 1:
                use_edge = True

            if include_single_patch_mesh_boundaries and num_groups == 1:
                use_edge = True

            if use_edge:
                candidates[edge_key] = {
                    "points": info["points"],
                    "groups": set(info["groups"]),
                    "source": "mesh_edge",
                }

    if include_obj_lines:
        for edge_key, info in parsed["line_tracker"].items():
            if edge_key not in candidates:
                candidates[edge_key] = {
                    "points": info["points"],
                    "groups": set(info["groups"]),
                    "source": "obj_line",
                }
            else:
                candidates[edge_key]["groups"].update(info["groups"])
                candidates[edge_key]["source"] += "+obj_line"

    return candidates


def build_patch_aligned_ribbon_mesh(
    parsed,
    width,
    normal_offset=0.0,
    include_inter_patch_edges=True,
    include_obj_lines=True,
    include_single_patch_mesh_boundaries=False,
    one_ribbon_per_incident_patch=True,
):
    group_normals = compute_group_normals(parsed)

    ribbon_vertices = []
    ribbon_faces = []
    ribbon_face_colors = []
    ribbon_records = []

    candidates = collect_feature_segments(
        parsed,
        include_inter_patch_edges=include_inter_patch_edges,
        include_obj_lines=include_obj_lines,
        include_single_patch_mesh_boundaries=include_single_patch_mesh_boundaries,
    )

    group_id2name = parsed["group_id2name"]
    groups = parsed["groups"]

    ribbon_id = 0

    for segment_id, (edge_key, seg) in enumerate(candidates.items()):
        p0, p1 = seg["points"]
        incident_gids = sorted(seg["groups"])

        if not incident_gids:
            incident_gids = [None]

        if not one_ribbon_per_incident_patch and incident_gids:
            incident_gids = incident_gids[:1]

        for gid in incident_gids:
            if gid is not None:
                patch_normal = get_local_patch_normal(
                    edge_key, gid, parsed, group_normals
                )
            else:
                patch_normal = fallback_patch_normal_from_segment(p0, p1)

            if patch_normal is None:
                patch_normal = fallback_patch_normal_from_segment(p0, p1)

            ribbon = make_patch_aligned_ribbon(
                p0=p0,
                p1=p1,
                patch_normal=patch_normal,
                width=width,
                normal_offset=normal_offset,
            )

            if ribbon is None:
                continue

            base = len(ribbon_vertices)
            face_base = len(ribbon_faces)

            ribbon_vertices.extend(ribbon["vertices"].tolist())
            ribbon_faces.extend((ribbon["faces"] + base).tolist())

            color = [220, 60, 60, 255]
            if gid is not None and gid in groups:
                color = groups[gid]["color"]

            ribbon_face_colors.extend([color, color])

            ribbon_records.append(
                {
                    "ribbon_id": ribbon_id,
                    "segment_id": segment_id,
                    "source": seg["source"],
                    "patch_id": None if gid is None else int(gid),
                    "patch_name": None if gid is None else group_id2name.get(gid),
                    "original_start": np.asarray(p0, dtype=float).tolist(),
                    "original_end": np.asarray(p1, dtype=float).tolist(),
                    "ribbon_vertex_indices": [base, base + 1, base + 2, base + 3],
                    "ribbon_face_indices": [face_base, face_base + 1],
                    "width": float(width),
                    "normal_offset": float(normal_offset),
                    "patch_normal": ribbon["patch_normal"].tolist(),
                    "ribbon_direction": ribbon["ribbon_direction"].tolist(),
                }
            )

            ribbon_id += 1

    if not ribbon_vertices:
        mesh = trimesh.Trimesh(
            vertices=np.zeros((0, 3)), faces=np.zeros((0, 3)), process=False
        )
        return mesh, ribbon_records

    mesh = trimesh.Trimesh(
        vertices=np.asarray(ribbon_vertices, dtype=float),
        faces=np.asarray(ribbon_faces, dtype=np.int64),
        process=False,
    )

    mesh.visual.face_colors = np.asarray(ribbon_face_colors, dtype=np.uint8)

    return mesh, ribbon_records


def build_original_patch_meshes(parsed):
    vertices = parsed["vertices"]
    groups = parsed["groups"]

    patch_meshes = []

    for gid, data in groups.items():
        if not data["faces"]:
            continue

        old_faces = np.asarray([f["indices"] for f in data["faces"]], dtype=np.int64)
        unique_indices = np.unique(old_faces.flatten())

        # Remap old global OBJ vertex indices to compact local mesh indices.
        idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_indices)}

        new_vertices = vertices[unique_indices]
        new_faces = np.asarray(
            [[idx_map[i] for i in tri] for tri in old_faces],
            dtype=np.int64,
        )

        mesh = trimesh.Trimesh(
            vertices=new_vertices,
            faces=new_faces,
            process=False,
        )

        mesh.visual.face_colors = np.asarray(
            [data["color"] for _ in range(len(new_faces))],
            dtype=np.uint8,
        )

        patch_meshes.append((gid, mesh))

    return patch_meshes


def export_patch_aligned_ribbons_for_trellis2(
    input_obj_path,
    output_prefix,
    width=None,
    width_fraction=0.004,
    normal_offset=None,
    normal_offset_fraction_of_width=0.0,
    include_original_surface=True,
    include_inter_patch_edges=True,
    include_obj_lines=True,
    include_single_patch_mesh_boundaries=False,
):
    """
    Exports:
        output_prefix + ".glb"
            Original patch mesh, optionally, plus patch-aligned ribbon mesh.

        output_prefix + "_ribbons_only.glb"
            Only the ribbon mesh.

        output_prefix + "_ribbons.json"
            Sidecar metadata for exact recovery before neural encoding.

    width:
        Absolute ribbon width. If None, uses width_fraction * bbox diagonal.

    normal_offset:
        Absolute offset along patch normal. If None, uses
        normal_offset_fraction_of_width * width.

        For pure geometric encoding, use 0.
        For visibility / Trellis2 survival, try 0.1 * width to 0.5 * width.
    """
    parsed = parse_obj_with_patch_info(input_obj_path)

    vertices = parsed["vertices"]
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_diag = float(np.linalg.norm(bbox_max - bbox_min))

    if width is None:
        width = width_fraction * bbox_diag

    if normal_offset is None:
        normal_offset = normal_offset_fraction_of_width * width

    ribbon_mesh, ribbon_records = build_patch_aligned_ribbon_mesh(
        parsed=parsed,
        width=width,
        normal_offset=normal_offset,
        include_inter_patch_edges=include_inter_patch_edges,
        include_obj_lines=include_obj_lines,
        include_single_patch_mesh_boundaries=include_single_patch_mesh_boundaries,
        one_ribbon_per_incident_patch=True,
    )

    # Export ribbons only.
    ribbon_mesh.export(output_prefix + "_ribbons_only.glb")

    # # Export sidecar metadata.
    # with open(output_prefix + "_ribbons.json", "w") as f:
    #     json.dump(
    #         {
    #             "input_obj_path": input_obj_path,
    #             "width": float(width),
    #             "normal_offset": float(normal_offset),
    #             "num_ribbons": len(ribbon_records),
    #             "ribbons": ribbon_records,
    #         },
    #         f,
    #         indent=2,
    #     )

    # Export combined scene.
    scene = trimesh.Scene()

    if include_original_surface:
        for gid, mesh in build_original_patch_meshes(parsed):
            name = parsed["group_id2name"].get(gid, f"Patch_{gid}")
            scene.add_geometry(mesh, node_name=f"Patch_{gid}_{name}")

    # scene.add_geometry(ribbon_mesh, node_name="PatchAlignedRibbons")
    scene.export(output_prefix + ".glb")

    print(f"Saved combined mesh:      {output_prefix}.glb")
    print(f"Saved ribbons only:       {output_prefix}_ribbons_only.glb")
    print(f"Saved ribbon sidecar:     {output_prefix}_ribbons.json")
    print(f"Ribbon width:             {width}")
    print(f"Normal offset:            {normal_offset}")
    print(f"Number of ribbon patches: {len(ribbon_records)}")

    return True, ribbon_mesh, ribbon_records
