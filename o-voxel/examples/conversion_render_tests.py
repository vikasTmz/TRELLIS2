import torch
import o_voxel
import utils
import trimesh
import glob
import numpy as np
import imageio
import utils3d


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
    cell_min = voxel_indices / res  # - 0.5  # cell min corner in normalized space

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

        # # ------------------------------------------------------------
        # # 2) Write edge endpoint vertices
        # # ------------------------------------------------------------
        # f.write("\no edge_endpoints\n")
        # for p in extra_vertices:
        #     f.write(f"v {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")

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


RES = 64

glb_paths = glob.glob(
    "/home/vthamizharas/Documents/TRELLIS.2/datasets/ObjaverseXL_sketchfab/raw/hf-objaverse-v1/glbs/**/*.glb"
)

for glb_path in glb_paths:
    # asset = utils.get_helmet()
    asset = trimesh.load(glb_path)  # , force="mesh")

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

    print(voxel_indices[0], dual_vertices[0], intersected[0])

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

    # # ------------------------------------------------------------
    # # 2) Export dual vertices as PLY point cloud
    # # ------------------------------------------------------------
    # export_dual_vertices_ply(
    #     dual_vertices,
    #     glb_path.replace(".glb", "_dualvertices.ply").split("/")[-1],
    # )

    # # ------------------------------------------------------------
    # # 3) Export intersected edges as GLB thin-cylinder mesh
    # # ------------------------------------------------------------
    # export_intersected_edges_glb(
    #     voxel_indices,
    #     intersected,
    #     RES,
    #     glb_path.replace(".glb", "_intersected_edges.ply").split("/")[-1],
    #     radius=0.01 / RES,
    # )

    # ------------------------------------------------------------
    # 4) Export dual vertices and intersected edges together as OBJ
    # ------------------------------------------------------------
    export_dual_vertices_and_intersected_edges_obj(
        dual_vertices=dual_vertices,
        voxel_indices=voxel_indices,
        intersected=intersected,
        res=RES,
        out_path=glb_path.replace(".glb", "_dual_vertices_and_edges.obj").split("/")[
            -1
        ],
    )
