import os
import sys

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "expandable_segments:True"  # Can save GPU memory
)
os.environ["ATTN_BACKEND"] = "xformers"
import cv2
import imageio
import numpy as np
from PIL import Image
import glob
import logging

import torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils, normal_based_segmentation
from trellis2.renderers import EnvMap
from trellis2.representations import Mesh, MeshWithVoxel
from trellis2.modules.sparse import SparseTensor
import o_voxel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Loaded modules")

from huggingface_hub import login

login(token=sys.argv[1])


# 2. Load Pipeline
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

# # Decode sparse structure latent
# ss_decoder = pipeline.models["sparse_structure_decoder"]

# print(type(ss_decoder))  # shows full type incl. module path
# print(ss_decoder.__class__.__name__)  # just the class name (e.g., "MyEncoder")
# print(ss_decoder.__class__.__module__)  # module it came from
# print(f"{ss_decoder.__class__.__module__}.{ss_decoder.__class__.__name__}")

logging.info("Pipeline loaded and moved to CUDA")

RUN_SS_DECODER = True

for shape_resolution in [512, 256]:
    ss_latents = sorted(
        glob.glob(
            f"datasets/ObjaverseXL_sketchfab/ss_latents/ss_enc_conv3d_16l8_fp16_64_{shape_resolution}/*.npz"
        )
    )
    shape_latents = sorted(
        glob.glob(
            f"datasets/ObjaverseXL_sketchfab/shape_latents/shape_enc_next_dc_f16c32_fp16_{shape_resolution}/*.npz"
        )
    )
    # 3. Load
    for file_ID in range(len(ss_latents)):
        ss_latent_path = ss_latents[file_ID]
        filename = os.path.splitext(os.path.basename(ss_latent_path))[0]
        shape_latent_path = shape_latents[file_ID]
        print(
            os.path.splitext(os.path.basename(ss_latent_path))[0],
            os.path.splitext(os.path.basename(shape_latent_path))[0],
        )
        # Loading outputs of encode_shape_latents.py (SC-VAE)
        enc_feats = torch.from_numpy(np.load(shape_latent_path)["feats"])
        enc_feats = enc_feats.cuda()
        # enc_coords = torch.from_numpy(np.load(shape_latent_path)["coords"])
        # enc_coords = enc_coords.cuda()
        # Loading outputs of encode_ss_latents.py (Sparse Strcture for generative model)
        # Specifically, the occupancy layout of the sparse voxel grid
        z_s = torch.from_numpy(np.load(ss_latent_path)["z"])
        z_s = z_s.unsqueeze(0).cuda()
        enc_feats.shape
        z_s.shape
        if RUN_SS_DECODER:
            filename += f"_{shape_resolution}_withSSDecoder"
        if RUN_SS_DECODER:
            resolution = 64
            if pipeline.low_vram:
                pipeline.models["sparse_structure_decoder"].to(pipeline.device)
            decoded = pipeline.models["sparse_structure_decoder"](z_s) > 0
            print(f"Decoded shape: {decoded.shape}")
            if pipeline.low_vram:
                pipeline.models["sparse_structure_decoder"].cpu()
            if resolution != decoded.shape[2]:
                ratio = decoded.shape[2] // resolution
                decoded = (
                    torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0)
                    > 0.5
                )
        if RUN_SS_DECODER:
            dec_coords = torch.argwhere(decoded)[
                :, [0, 2, 3, 4]
            ].int()  # .to(dtype=torch.int32)
            # coords = coords[:, 1:]  # remove batch dimension
            dec_coords = dec_coords.contiguous()
            print(dec_coords.shape, dec_coords.dtype, dec_coords.device)
        else:
            dec_coords = enc_coords.int().contiguous()
            if dec_coords.shape[1] == 3:
                dec_coords = torch.cat(
                    [
                        torch.zeros(
                            dec_coords.shape[0],
                            1,
                            dtype=dec_coords.dtype,
                            device=dec_coords.device,
                        ),
                        dec_coords,
                    ],
                    dim=1,
                )
            print(dec_coords.shape, dec_coords.dtype, dec_coords.device)
        # SS_DECODER doesn't decode features, so we use the encoded features
        # these features come from a generative model structured_latent_flow.py
        shape_slat = SparseTensor(
            feats=enc_feats.to(dec_coords.device),
            coords=dec_coords,
        )
        shape_slat.shape, shape_slat.feats.shape, shape_slat.coords.shape, shape_slat.device
        meshes, subs = pipeline.decode_shape_slat(shape_slat, shape_resolution)
        # if pipeline.low_vram:
        #     shape_decoder.to(pipeline.device)
        #     shape_decoder.low_vram = True
        # meshes, subs = shape_decoder(shape_slat, return_subs=True)
        # if pipeline.low_vram:
        #     shape_decoder.cpu()
        #     shape_decoder = False

        mesh = meshes[0]
        mesh.fill_holes()
        mesh.simplify(16777216)
        print(f"Vertices: {mesh.vertices.shape}, Faces: {mesh.faces.shape}")
        # 1. Move to CPU and convert to numpy
        # Ensure vertices are float32 and faces are int32
        v = mesh.vertices.detach().cpu().numpy().astype(np.float32)
        f = mesh.faces.detach().cpu().numpy().astype(np.int32)
        with open(
            f"datasets/ObjaverseXL_sketchfab/decode_shapes/trellis2_vae_outputs/{filename}.ply",
            "wb",
        ) as f_out:
            # 2. Write ASCII Header
            # We define face list size as 'int' to allow fast vectorized writing
            header = (
                f"ply\n"
                f"format binary_little_endian 1.0\n"
                f"element vertex {v.shape[0]}\n"
                f"property float x\n"
                f"property float y\n"
                f"property float z\n"
                f"element face {f.shape[0]}\n"
                f"property list int int vertex_indices\n"
                f"end_header\n"
            )
            f_out.write(header.encode("ascii"))
            # 3. Write Vertices (Direct Binary Dump)
            f_out.write(v.tobytes())
            # 4. Write Faces (Direct Binary Dump)
            # PLY faces need a 'count' before indices.
            # We stack a column of 3s to get [3, v1, v2, v3]
            # Since we defined the count type as 'int' in the header,
            # we can dump this strictly as a uniform int32 array.
            f_padding = np.ones((f.shape[0], 1), dtype=np.int32) * 3
            faces_data = np.hstack((f_padding, f))
            f_out.write(faces_data.tobytes())
