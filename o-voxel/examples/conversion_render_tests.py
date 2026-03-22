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


def export_dual_grid_visualization(
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
            # edge_is_green[canon_edge(c000, c100)] = True
            edge_is_green[canon_edge(c011, c111)] = True
        if flags[1]:
            # edge_is_green[canon_edge(c000, c010)] = True
            edge_is_green[canon_edge(c101, c111)] = True
        if flags[2]:
            # edge_is_green[canon_edge(c000, c001)] = True
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


def export_dual_vertices_ply(dual_vertices, out_path):
    dual_vertices_np = dual_vertices.cpu().numpy()
    colors_np = np.clip((dual_vertices_np + 0.5) * 255, 0, 255).astype(np.uint8)

    pc = trimesh.points.PointCloud(
        vertices=dual_vertices_np,
        colors=colors_np,
    )
    pc.export(out_path)


def export_intersected_edges_glb(
    voxel_indices, intersected, res, out_path, radius=None
):
    """
    Export intersected voxel-grid edges as a thin-cylinder mesh in GLB format.

    Assumes voxel_indices are the minimum corner indices of each cell, which is
    consistent with the packing formula:
        dual_vertices * RES - voxel_indices
    """
    if radius is None:
        radius = 0.08 / res  # thin but visible

    if intersected.dtype != torch.bool:
        intersected = intersected > 0

    voxel_indices = voxel_indices.float().cpu()
    intersected = intersected.cpu()

    step = 1.0 / res
    cell_min = voxel_indices / res - 0.5  # cell min corner in normalized space

    axis_vecs = torch.tensor(
        [
            [step, 0.0, 0.0],  # x-edge
            [0.0, step, 0.0],  # y-edge
            [0.0, 0.0, step],  # z-edge
        ],
        dtype=torch.float32,
    )

    axis_colors = np.array(
        [
            [255, 80, 80, 255],  # x = red
            [80, 255, 80, 255],  # y = green
            [80, 80, 255, 255],  # z = blue
        ],
        dtype=np.uint8,
    )

    meshes = []

    for axis in range(3):
        mask = intersected[:, axis]
        if not mask.any():
            continue

        p0s = cell_min[mask].numpy()
        p1s = (cell_min[mask] + axis_vecs[axis]).numpy()

        for p0, p1 in zip(p0s, p1s):
            cyl = trimesh.creation.cylinder(
                radius=radius,
                segment=np.stack([p0, p1], axis=0),
                sections=6,  # lightweight
            )
            cyl.visual.face_colors = axis_colors[axis]
            meshes.append(cyl)

    if len(meshes) == 0:
        print(f"No intersected edges found for {out_path}")
        return

    edge_mesh = trimesh.util.concatenate(meshes)
    edge_mesh.export(out_path)

    meshes = []

    for axis in range(9):

        p0s = cell_min[mask].numpy()
        p1s = (cell_min[mask] + axis_vecs[axis]).numpy()

        for p0, p1 in zip(p0s, p1s):
            cyl = trimesh.creation.cylinder(
                radius=radius,
                segment=np.stack([p0, p1], axis=0),
                sections=6,  # lightweight
            )
            cyl.visual.face_colors = axis_colors[axis]
            meshes.append(cyl)

    if len(meshes) == 0:
        print(f"No intersected edges found for {out_path}")
        return

    edge_mesh = trimesh.util.concatenate(meshes)
    edge_mesh.export(out_path.replace("_intersected_edges", "_voxel"))


def export_dual_vertices_and_intersected_edges_obj(
    dual_vertices,
    voxel_indices,
    intersected,
    res,
    out_path,
):
    """
    Export a single OBJ containing:
      - dual vertices as OBJ point primitives (`p`)
      - intersected voxel-grid edges as OBJ line primitives (`l`)

    Everything is written in the same normalized coordinate space.
    """

    if intersected.dtype != torch.bool:
        intersected = intersected > 0

    dual_vertices = dual_vertices.detach().cpu().float().numpy()
    voxel_indices = voxel_indices.detach().cpu().float()
    intersected = intersected.detach().cpu()

    step = 1.0 / res
    cell_min = voxel_indices / res  # - 0.5  # normalized object space

    axis_vecs = torch.tensor(
        [
            [step, 0.0, 0.0],  # x-edge
            [0.0, step, 0.0],  # y-edge
            [0.0, 0.0, step],  # z-edge
        ],
        dtype=torch.float32,
    )

    # We will write:
    #   1) dual-vertex positions first
    #   2) extra vertices for line endpoints
    #   3) point primitives for dual vertices
    #   4) line primitives for edges
    extra_vertices = []
    edge_lines_by_axis = [[], [], []]  # store OBJ 1-based index pairs

    dual_count = len(dual_vertices)

    for axis in range(3):
        mask = intersected[:, axis]
        if not mask.any():
            continue

        p0s = cell_min[mask].numpy()
        p1s = (cell_min[mask] + axis_vecs[axis]).numpy()

        for p0, p1 in zip(p0s, p1s):
            i0 = dual_count + len(extra_vertices) + 1
            extra_vertices.append(p0)

            i1 = dual_count + len(extra_vertices) + 1
            extra_vertices.append(p1)

            edge_lines_by_axis[axis].append((i0, i1))

    with open(out_path, "w") as f:
        f.write("# dual vertices + intersected edges\n")
        f.write("# dual vertices are OBJ point primitives\n")
        f.write("# intersected edges are OBJ line primitives\n\n")

        # ------------------------------------------------------------
        # 1) Write dual vertices
        # ------------------------------------------------------------
        f.write("o dual_vertices\n")
        for p in dual_vertices:
            f.write(f"v {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")

        # ------------------------------------------------------------
        # 2) Write edge endpoint vertices
        # ------------------------------------------------------------
        f.write("\no edge_endpoints\n")
        for p in extra_vertices:
            f.write(f"v {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")

        # ------------------------------------------------------------
        # 3) Emit dual vertices as point primitives
        # ------------------------------------------------------------
        f.write("\ng dual_vertices\n")
        for i in range(1, dual_count + 1):
            f.write(f"p {i}\n")

        # ------------------------------------------------------------
        # 4) Emit intersected edges as line primitives
        # ------------------------------------------------------------
        axis_names = ["x_edges", "y_edges", "z_edges"]
        for axis in range(3):
            f.write(f"\ng {axis_names[axis]}\n")
            for i0, i1 in edge_lines_by_axis[axis]:
                f.write(f"l {i0} {i1}\n")


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


RES = 16

# glb_paths = glob.glob(
#     "/home/vthamizharas/Documents/TRELLIS.2/datasets/ObjaverseXL_sketchfab/raw/hf-objaverse-v1/glbs/**/*.glb"
# )
glb_paths = glob.glob("shapes/*.glb")

for glb_path in glb_paths:
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

    export_dual_grid_visualization(
        voxel_indices=voxel_indices,
        dual_vertices=dual_vertices,
        intersected=intersected,
        res=RES,
        out_path=glb_path.replace(".glb", "_dual_grid_debug.glb").split("/")[-1],
    )

    mesh.export(glb_path.replace(".glb", "_original_mesh.obj").split("/")[-1])
