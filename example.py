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
from trellis2.utils import render_utils, normal_based_segmentation
from trellis2.renderers import EnvMap
import o_voxel

import glob
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Loaded modules")

INPUT_TYPE = "image_variations"
input_imgs = sorted(glob.glob(f"assets/{INPUT_TYPE}/*"))
print(input_imgs)

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


for img_path in input_imgs:
    try:
        # 3. Load Image & Run
        image = Image.open(img_path)
        filename = os.path.splitext(os.path.basename(img_path))[0]

        mesh = pipeline.run(image, filename, num_samples=1)
        logging.info(f"mesh variable size {len(mesh)}")
        mesh = mesh[0]
        mesh.simplify(16777216)  # nvdiffrast limit

        logging.info("Image processed and 3D mesh generated")

        # 4. Render Video
        trellis_video = render_utils.make_pbr_vis_frames(
            render_utils.render_video(mesh, envmap=envmap, num_frames=70), input=image
        )

        imageio.mimsave(
            f"output/{INPUT_TYPE}/Render_{filename}.mp4", trellis_video, fps=15
        )

        logging.info(f"Video rendered and saved as Render_{filename}.mp4")

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

        glb.export(f"output/{INPUT_TYPE}/{filename}.glb")  # , extension_webp=True

        logging.info(f"GLB exported as {filename}.glb")

    except Exception as e:
        logging.error(f"Error processing {img_path}: {e}")

    break
