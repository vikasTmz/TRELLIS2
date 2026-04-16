"""
Turntable‐style offscreen rendering of a trimesh object with pyrender.
Renders N frames rotating the object around its Y axis.
"""

import os
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import glob
import cv2

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


def offscreen_render_turntable(mesh, filename):
    # try:

    mesh = mesh.copy()
    mesh.apply_scale(0.15)
    mesh = Mesh.from_trimesh(mesh, smooth=False)

    # Base pose of the tripoSG (you can tweak this)
    base_pose = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, -0.1],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    # ==============================================================================
    # Lighting
    # ==============================================================================

    # A directional light and a spot light
    direc_l = DirectionalLight(color=np.ones(3), intensity=1.0)
    spot_l = SpotLight(
        color=np.ones(3),
        intensity=1.0,
        innerConeAngle=np.pi / 16,
        outerConeAngle=np.pi / 6,
    )

    # ==============================================================================
    # Camera
    # ==============================================================================

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
    # cam_pose = np.array([
    #     [0.0,  -np.sqrt(2)/2, np.sqrt(2)/2, 0.5],
    #     [1.0, 0.0,           0.0,           0.0],
    #     [0.0,  np.sqrt(2)/2,  np.sqrt(2)/2, 0.4],
    #     [0.0,  0.0,           0.0,          1.0]
    # ])

    # ==============================================================================
    # Scene setup
    # ==============================================================================
    scene = Scene(
        ambient_light=np.array([0.2, 0.2, 0.2, 1.0]), bg_color=[0.0, 0.0, 0.1, 1.0]
    )
    node = scene.add(mesh, pose=base_pose)
    _ = scene.add(direc_l, pose=cam_pose)
    _ = scene.add(spot_l, pose=cam_pose)
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
    num_frames = 120

    # Compute object center in world coords (here we use the translation part of base_pose)
    object_center = base_pose[:3, 3]

    # # Ensure output directory exists
    # out_dir = "turntable_frames"
    # os.makedirs(out_dir, exist_ok=True)

    # for i, angle in enumerate(np.linspace(0, 2*np.pi, num_frames, endpoint=False)):
    axis_vector = [0, 0.4, 1]
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
        color, depth = r.render(scene)
        # Save frame
        combined.append(color)

    # poses = [
    #     [[0, 0, 1], [np.pi / 4, np.pi * 3 / 4]],
    #     # [[0, 0, 1], [np.pi, 3 * np.pi / 2]],
    #     [[0, 1, 0], [np.pi / 2, -np.pi / 2]],
    # ]

    # combined = None
    # for pose in poses:
    #     # print(f"Rendering pose with axis {pose[0]} and angles {pose[1]}...")
    #     axis_vector, angles = pose
    #     row_img = None
    #     for angle in angles:
    #         # Build a rotation matrix around Y axis centered on object_center
    #         R = trimesh.transformations.rotation_matrix(
    #             angle, axis_vector, point=object_center
    #         )
    #         # Apply rotation to the base pose
    #         new_pose = R.dot(base_pose)
    #         # Update the node’s transform
    #         node.matrix = new_pose

    #         # Render
    #         color, depth = r.render(scene)
    #         # Save frame
    #         # fname = os.path.join(out_dir, f'frame_{i:03d}.png')
    #         if row_img is None:
    #             row_img = color
    #         else:
    #             row_img = np.concatenate((row_img, color), axis=1)

    #     if combined is None:
    #         combined = row_img
    #     else:
    #         combined = np.concatenate((combined, row_img), axis=0)

    # print(f"Saving result to {filename}...")

    # plt.imsave(filename, combined)

    # Clean up
    r.delete()

    return combined

    # except Exception as e:
    #     print(f"Error processing {filename}: {e}")
    #     continue


def offscreen_render_fixedviews(mesh, filename):
    # try:

    # mesh = mesh.copy()
    mesh.apply_scale(0.5)
    mesh = Mesh.from_trimesh(mesh, smooth=False)

    # Base pose of the tripoSG (you can tweak this)
    base_pose = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, -0.1],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    # ==============================================================================
    # Lighting
    # ==============================================================================

    # A directional light and a spot light
    direc_l = DirectionalLight(color=np.ones(3), intensity=1.0)
    spot_l = SpotLight(
        color=np.ones(3),
        intensity=1.0,
        innerConeAngle=np.pi / 16,
        outerConeAngle=np.pi / 6,
    )

    # ==============================================================================
    # Camera
    # ==============================================================================

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
    # cam_pose = np.array([
    #     [0.0,  -np.sqrt(2)/2, np.sqrt(2)/2, 0.5],
    #     [1.0, 0.0,           0.0,           0.0],
    #     [0.0,  np.sqrt(2)/2,  np.sqrt(2)/2, 0.4],
    #     [0.0,  0.0,           0.0,          1.0]
    # ])

    # ==============================================================================
    # Scene setup
    # ==============================================================================
    scene = Scene(
        ambient_light=np.array([0.2, 0.2, 0.2, 1.0]), bg_color=[0.0, 0.0, 0.1, 1.0]
    )
    node = scene.add(mesh, pose=base_pose)
    _ = scene.add(direc_l, pose=cam_pose)
    _ = scene.add(spot_l, pose=cam_pose)
    cam_node = scene.add(cam, pose=cam_pose)

    # ==============================================================================
    # Offscreen renderer
    # ==============================================================================

    width, height = 512 * 2, 512 * 2
    r = OffscreenRenderer(viewport_width=width, viewport_height=height)

    # Compute object center in world coords (here we use the translation part of base_pose)
    object_center = base_pose[:3, 3]

    # # Ensure output directory exists
    # out_dir = "turntable_frames"
    # os.makedirs(out_dir, exist_ok=True)

    poses = [
        [[0, 0, 1], [np.pi / 4, np.pi * 3 / 4]],
        # [[0, 0, 1], [np.pi, 3 * np.pi / 2]],
        # [[0, 1, 0], [np.pi / 2, -np.pi / 2]],
    ]

    combined = None
    for pose in poses:
        # print(f"Rendering pose with axis {pose[0]} and angles {pose[1]}...")
        axis_vector, angles = pose
        row_img = None
        for angle in angles:
            # Build a rotation matrix around Y axis centered on object_center
            R = trimesh.transformations.rotation_matrix(
                angle, axis_vector, point=object_center
            )
            # Apply rotation to the base pose
            new_pose = R.dot(base_pose)
            # Update the node’s transform
            node.matrix = new_pose

            # Render
            color, depth = r.render(scene)
            # Save frame
            # fname = os.path.join(out_dir, f'frame_{i:03d}.png')
            if row_img is None:
                row_img = color
            else:
                row_img = np.concatenate((row_img, color), axis=1)

        if combined is None:
            combined = row_img
        else:
            combined = np.concatenate((combined, row_img), axis=0)

    # print(f"Saving result to {filename}...")

    plt.imsave(filename, combined)

    # Clean up
    r.delete()

    return combined

    # except Exception as e:
    #     print(f"Error processing {filename}: {e}")
    #     continue
