"""
Turntable‐style offscreen rendering of a trimesh object with pyrender.
Renders N frames rotating the object around its Y axis.
"""

import os
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import glob
import imageio

os.environ["PYOPENGL_PLATFORM"] = (
    "egl"  # must come before importing pyrender, OpenGL, etc.
)

# --- pyrender imports must come after pyglet option set ---
import pyglet

pyglet.options["shadow_window"] = False

from pyrender import (
    PerspectiveCamera,
    DirectionalLight,
    SpotLight,
    PointLight,
    Mesh,
    Scene,
    Viewer,
    OffscreenRenderer,
)
import pyrender.constants as pyrc


def offscreen_render_turntable(mesh_path):
    if mesh_path.endswith(".ply"):
        mesh = trimesh.load(mesh_path, process=False)
    elif mesh_path.endswith(".glb"):
        with open(mesh_path, "rb") as f:
            f.seek(0)  # IMPORTANT if the handle was used earlier
            scene = trimesh.load(
                file_obj=f, file_type="glb", force="scene", process=False
            )
        mesh = scene.dump(concatenate=True)
    mesh.apply_scale(0.003)  # 0.004
    center = mesh.bounds.mean(axis=0)  # (min+max)/2
    mesh.apply_translation(-center)
    # mesh.rezero()
    # mesh.fix_normals()
    mesh = Mesh.from_trimesh(mesh, smooth=True)
    base_pose = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, -0.1],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    direc_l = DirectionalLight(color=np.ones(3), intensity=1.0)
    spot_l = SpotLight(
        color=np.ones(3),
        intensity=1.0,
        innerConeAngle=np.pi / 16,
        outerConeAngle=np.pi / 6,
    )
    cam = PerspectiveCamera(yfov=(np.pi / 4.0))
    cam_pose = np.array(
        [
            #  right_x  up_x   localZ_x   cam_x
            [0.0, 0.0, 1.0, 0.8],
            #  right_y  up_y   localZ_y   cam_y
            [1.0, 0.0, 0.0, 0.0],
            #  right_z  up_z   localZ_z   cam_z
            [0.0, 1.0, 0.0, -0.1],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    scene = Scene(
        ambient_light=np.array([0.8, 0.3, 0.88, 1.0]), bg_color=[1.0, 1.0, 1.0, 1.0]
    )
    node = scene.add(mesh, pose=base_pose)
    _ = scene.add(direc_l, pose=cam_pose)
    # _ = scene.add(spot_l, pose=cam_pose)
    cam_node = scene.add(cam, pose=cam_pose)
    # ==============================================================================
    # Offscreen renderer
    # ==============================================================================
    width, height = 512 * 2, 512 * 2
    r = OffscreenRenderer(viewport_width=width, viewport_height=height)
    # ==============================================================================
    # Turntable loop
    # ==============================================================================
    # Number of frames around the full 360° (you can increase for smoother animation)
    num_frames = 60
    # Compute object center in world coords (here we use the translation part of base_pose)
    object_center = base_pose[:3, 3]
    axis_vector = [0, 0.6, 1]
    combined = []
    for i, angle in enumerate(np.linspace(0, 2 * np.pi, num_frames, endpoint=False)):
        # Build a rotation matrix around Y axis centered on object_center
        R = trimesh.transformations.rotation_matrix(
            angle, axis_vector, point=object_center
        )
        # Apply rotation to the base pose
        new_pose = R.dot(base_pose)
        # Update the node’s transform
        node.matrix = new_pose
        # Render
        color, depth = r.render(scene, flags=pyrc.RenderFlags.SKIP_CULL_FACES)
        # Save frame
        combined.append(color)
    r.delete()
    return combined


import glob

plys = glob.glob("../ground_truth/gt_*.glb")
# plys = glob.glob("*.ply")

for mesh_path in plys:
    try:
        frames = offscreen_render_turntable(mesh_path)
        video_path = mesh_path.replace(".glb", ".mp4")
        imageio.mimsave(video_path, frames, fps=15)
    except Exception as e:
        print(f"Error processing {mesh_path}: {e}")
