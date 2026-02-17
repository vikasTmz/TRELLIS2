import bpy
import os
from mathutils import Matrix

# ---------------- USER SETTINGS ----------------
OUTPATH = bpy.path.abspath(
    "/Users/vthamizh/Documents/Research/Diffusion_models/Autodesk_BrepGen/output_trellis2/encode_decode_tests/pcb_vise_segment4.obj"
)  # next to .blend
COLOR_ATTR_NAME = "FaceColor"  # your color attribute name
APPLY_TRANSFORMS = True  # True = world-space export, False = local-space
COLOR_ROUNDING = 4  # rounding for grouping (stabilizes float comparisons)
# ------------------------------------------------


def ensure_dir(filepath: str):
    folder = os.path.dirname(filepath)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)


def get_color_layer(mesh, name: str):
    """
    Returns (layer, domain) where layer.data[i].color exists.
    domain is 'CORNER' or 'POINT' when available.
    """
    # Blender 3.2+/4.x
    if hasattr(mesh, "color_attributes") and mesh.color_attributes:
        layer = mesh.color_attributes.get(name)
        if layer is not None:
            return layer, getattr(layer, "domain", "CORNER")
    # Legacy (older 2.9x/3.0-style)
    if hasattr(mesh, "vertex_colors") and mesh.vertex_colors:
        layer = mesh.vertex_colors.get(name)
        if layer is not None:
            # vertex_colors are loop/corner data
            return layer, "CORNER"
    return None, None


def poly_color_key(mesh, poly, layer, domain: str, rounding: int):
    """
    Computes a representative color for a polygon and returns a hashable key.
    - CORNER: average loop colors of the polygon
    - POINT : average vertex colors of the polygon
    """
    if layer is None:
        return ("no_color",)
    r = g = b = a = 0.0
    n = 0
    if domain == "POINT":
        # data indexed by vertex index
        for vidx in poly.vertices:
            c = layer.data[vidx].color
            r += c[0]
            g += c[1]
            b += c[2]
            a += c[3] if len(c) > 3 else 1.0
            n += 1
    else:
        # CORNER/loop domain: data indexed by loop index
        start = poly.loop_start
        end = start + poly.loop_total
        for li in range(start, end):
            c = layer.data[li].color
            r += c[0]
            g += c[1]
            b += c[2]
            a += c[3] if len(c) > 3 else 1.0
            n += 1
    if n == 0:
        return ("no_color",)
    r /= n
    g /= n
    b /= n
    a /= n
    return (
        round(r, rounding),
        round(g, rounding),
        round(b, rounding),
        round(a, rounding),
    )


def export_obj_with_facecolor_groups(
    obj,
    filepath: str,
    color_attr_name: str,
    apply_transforms: bool = True,
    rounding: int = 4,
):
    if not obj or obj.type != "MESH":
        raise RuntimeError("Active object must be a mesh.")
    ensure_dir(filepath)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    mesh = obj_eval.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    try:
        # Prepare geometry info
        # mesh.calc_normals()
        # mesh.calc_loop_triangles()
        # Color layer for grouping
        color_layer, color_domain = get_color_layer(mesh, color_attr_name)
        # Map polygon -> group id based on (rounded) face color
        color_to_gid = {}
        poly_gid = [0] * len(mesh.polygons)
        next_gid = 0
        for poly in mesh.polygons:
            key = poly_color_key(mesh, poly, color_layer, color_domain, rounding)
            gid = color_to_gid.get(key)
            if gid is None:
                gid = next_gid
                color_to_gid[key] = gid
                next_gid += 1
            poly_gid[poly.index] = gid
        # Organize triangles by their original polygon index
        tris_by_poly = [[] for _ in range(len(mesh.polygons))]
        for tri in mesh.loop_triangles:
            tris_by_poly[tri.polygon_index].append(tri.vertices)
        # Transforms
        if apply_transforms:
            M = obj.matrix_world
            N = M.to_3x3().inverted().transposed()  # correct normal transform
        else:
            M = Matrix.Identity(4)
            N = Matrix.Identity(3)
        # Write OBJ
        with open(filepath, "w", newline="\n") as f:
            f.write("# OBJ exported with face-color-based groups\n")
            f.write(f"o {obj.name}\n")
            # v
            for v in mesh.vertices:
                co = (M @ v.co.to_4d()).to_3d() if apply_transforms else v.co
                f.write("v {:.15g} {:.15g} {:.15g}\n".format(co.x, co.y, co.z))
            # vn (one per vertex; indices match vertex indices)
            for v in mesh.vertices:
                no = (N @ v.normal).normalized() if apply_transforms else v.normal
                f.write("vn {:.15g} {:.15g} {:.15g}\n".format(no.x, no.y, no.z))
            # Faces section: per original polygon, write group header then its triangles
            for p_idx, tri_list in enumerate(tris_by_poly):
                if not tri_list:
                    continue
                gid = poly_gid[p_idx]
                f.write(f"g face {gid}\n")
                for a, b, c in tri_list:
                    # OBJ is 1-based; normals use same index as vertices: v//vn
                    a1, b1, c1 = a + 1, b + 1, c + 1
                    f.write(f"f {a1}//{a1} {b1}//{b1} {c1}//{c1}\n")
            # Edges section: per polygon, output its boundary edges as 'l'
            for poly in mesh.polygons:
                verts = list(poly.vertices)
                if len(verts) < 2:
                    continue
                gid = poly_gid[poly.index]
                f.write(f"g edges face {gid}\n")
                for i in range(len(verts)):
                    a = verts[i] + 1
                    b = verts[(i + 1) % len(verts)] + 1
                    f.write(f"l {a} {b}\n")
        print(f"Wrote OBJ: {filepath}")
        print(f"Used color attribute: {color_attr_name} (domain={color_domain})")
        print(f"Unique color groups: {len(color_to_gid)}")
    finally:
        obj_eval.to_mesh_clear()


# ---- Run on active object ----
export_obj_with_facecolor_groups(
    bpy.context.active_object,
    OUTPATH,
    COLOR_ATTR_NAME,
    apply_transforms=APPLY_TRANSFORMS,
    rounding=COLOR_ROUNDING,
)
