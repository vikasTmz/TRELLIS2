import bpy
import bmesh
from mathutils import Color

# ---- settings ----
COLOR_ATTR_NAME = "FaceColor"
TARGET_COLORS = [
    (1.0, 1.0, 1.0, 1.0),
    (1.0, 0.0, 1.0, 1.0),
    (0.5, 1.0, 1.0, 1.0),
    (1.0, 0.5, 1.0, 1.0),
    (0.8, 0.6, 0.5, 1.0),
    (0.5, 0.6, 0.0, 1.0),
    (0.0, 1.0, 1.0, 1.0),
    (1.0, 0.0, 0.0, 1.0),
    (1.0, 1.0, 0.0, 1.0),
    (1.0, 1.0, 0.0, 1.0),
    (0.1, 0.3, 0.3, 1.0),
    (0.6, 0.0, 0.8, 1.0),
    (0.6394267984578837, 0.025010755222666936, 0.27502931836911926, 1.0),
    (0.22321073814882275, 0.7364712141640124, 0.6766994874229113, 1.0),
    (0.8921795677048454, 0.08693883262941615, 0.4219218196852704, 1.0),
    (0.029797219438070344, 0.21863797480360336, 0.5053552881033624, 1.0),
    (0.026535969683863625, 0.1988376506866485, 0.6498844377795232, 1.0),
    (0.5449414806032167, 0.2204406220406967, 0.5892656838759087, 1.0),
    (0.8094304566778266, 0.006498759678061017, 0.8058192518328079, 1.0),
    (0.6981393949882269, 0.3402505165179919, 0.15547949981178155, 1.0),
    (0.9572130722067812, 0.33659454511262676, 0.09274584338014791, 1.0),
    (0.09671637683346401, 0.8474943663474598, 0.6037260313668911, 1.0),
    (0.8071282732743802, 0.7297317866938179, 0.5362280914547007, 1.0),
    (0.9731157639793706, 0.3785343772083535, 0.552040631273227, 1.0),
    (0.8294046642529949, 0.6185197523642461, 0.8617069003107772, 1.0),
    (0.577352145256762, 0.7045718362149235, 0.045824383655662215, 1.0),
    (0.22789827565154686, 0.28938796360210717, 0.0797919769236275, 1.0),
    (0.23279088636103018, 0.10100142940972912, 0.2779736031100921, 1.0),
    (0.6356844442644002, 0.36483217897008424, 0.37018096711688264, 1.0),
    (0.2095070307714877, 0.26697782204911336, 0.936654587712494, 1.0),
    (0.6480353852465935, 0.6091310056669882, 0.171138648198097, 1.0),
    (0.7291267979503492, 0.1634024937619284, 0.3794554417576478, 1.0),
]
TARGET_COLOR = TARGET_COLORS[1]
# ------------------

obj = bpy.context.active_object
if not obj or obj.type != "MESH":
    raise RuntimeError("Active object must be a mesh.")

me = obj.data

# Ensure we are in Edit Mode so we can read selected faces from bmesh
if bpy.context.mode != "EDIT_MESH":
    raise RuntimeError("Must be in Edit Mode with faces selected.")

bm = bmesh.from_edit_mesh(me)
bm.faces.ensure_lookup_table()
bm.loops.ensure_lookup_table()

# Create or get the color attribute (Blender 3.2+ uses color_attributes)
# Store as CORNER so each face corner (loop) can have color (common for face coloring).
if hasattr(me, "color_attributes"):
    col_attr = me.color_attributes.get(COLOR_ATTR_NAME)
    if col_attr is None:
        col_attr = me.color_attributes.new(
            name=COLOR_ATTR_NAME, type="BYTE_COLOR", domain="CORNER"  # or 'FLOAT_COLOR'
        )
    color_layer = bm.loops.layers.color.get(COLOR_ATTR_NAME)
    if color_layer is None:
        color_layer = bm.loops.layers.color.new(COLOR_ATTR_NAME)
else:
    # Older Blender (2.9x / 3.0-ish): vertex_colors
    vcol = me.vertex_colors.get(COLOR_ATTR_NAME)
    if vcol is None:
        vcol = me.vertex_colors.new(name=COLOR_ATTR_NAME)
    color_layer = bm.loops.layers.color.get(COLOR_ATTR_NAME)
    if color_layer is None:
        color_layer = bm.loops.layers.color.new(COLOR_ATTR_NAME)

# Assign color to all loops of selected faces
count = 0
for f in bm.faces:
    if f.select:
        count += 1
        for loop in f.loops:
            print(loop[color_layer])
            loop[color_layer] = TARGET_COLOR

print(f"SELECT FACES count {count}")

bmesh.update_edit_mesh(me, loop_triangles=False, destructive=False)

print(
    f"Assigned color {TARGET_COLOR} to selected faces in '{obj.name}' on '{COLOR_ATTR_NAME}'."
)
