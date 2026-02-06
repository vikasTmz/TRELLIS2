import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "expandable_segments:True"  # Can save GPU memory
)
os.environ["ATTN_BACKEND"] = "xformers"
import cv2
import imageio
import numpy as np
from PIL import Image
import torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils, normal_based_segmentation, offline_render
from trellis2.renderers import EnvMap

import o_voxel

import glob
import logging
import pymeshlab as ml

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Loaded modules")

input_imgs = glob.glob("assets/autobrep_variants/*.png")

# 1. Setup Environment Map
envmap = EnvMap(
    torch.tensor(
        cv2.cvtColor(
            cv2.imread("assets/hdri/forest.exr", cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB,
        ),
        dtype=torch.float32,
        device="cuda",
    )
)

logging.info("Environment map set up")

# 2. Load Pipeline
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

logging.info("Pipeline loaded and moved to CUDA")

#####################

# img_path = "assets/thingi10k_variants/1_gpt_1.jpg"

for img_path in input_imgs:
    try:
        logging.info(f"Processing image: {img_path}")
        # 3. Load Image & Run
        image = Image.open(img_path)
        filename = os.path.splitext(os.path.basename(img_path))[0]
        mesh = pipeline.run(image)[0]
        mesh.simplify(16777216)  # nvdiffrast limit
        logging.info("Image processed and 3D mesh generated")
        # 4. Render Video
        trellis_result = render_video(mesh, envmap=envmap)
        trellis_video = make_pbr_vis_frames(trellis_result, input=image)
        # 5. Export to GLB
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=1000000,
            texture_size=4096,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=True,
        )
        glb.export(f"output/{filename}.glb")  # , extension_webp=True
        ms = ml.MeshSet()
        ms.load_new_mesh(f"output/{filename}.glb")
        # Save only vertices, faces, and normals (no colors, UVs, materials)
        ms.save_current_mesh(
            f"output/{filename}.obj",
            save_vertex_normal=True,  # write vn
            save_textures=False,  # no textures
            save_vertex_color=False,  # no per-vertex color
            save_wedge_texcoord=False,  # no vt/UVs
        )
        print("Saved OBJ")
        # 6. Segment and Render Segments
        video_2 = None
        for angle_deg in [5, 7, 15]:
            _, colored_mesh = normal_based_segmentation.segment_mesh_by_normal(
                normal_based_segmentation.strip_colors_and_uv(
                    normal_based_segmentation.load_glb_as_mesh(f"output/{filename}.obj")
                ),
                angle_deg=angle_deg,
                use_abs_dot=True,
                min_faces=1000,
                recolor_mesh=True,
            )
            seg_vid = offline_render.offscreen_render_turntable(
                colored_mesh,
                f"/home/vthamizharas/Documents/TRELLIS.2/output/segments/{filename}_angle={str(angle_deg)}_minfaces={str(1000)}.jpg",
            )
            if video_2 is None:
                video_2 = seg_vid
            else:
                video_2 = np.concatenate((video_2, seg_vid), axis=1)
        video = []
        for f1, f2 in zip(trellis_video, video_2):
            f2 = np.transpose(f2, (1, 0, 2))
            row = np.concatenate([f1, f2], axis=0)
            video.append(row)
        imageio.mimsave(
            f"output/{filename}.mp4",
            video,
            fps=15,
        )
        logging.info("Video rendered and saved as sample.mp4")
        # os.system(f"rm output/{filename}.glb")  # Clean up intermediate GLB file
        # os.system(f"rm output/{filename}.obj")
    except Exception as e:
        logging.error(f"Error processing {img_path}: {e}")
