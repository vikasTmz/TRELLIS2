#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import json
from huggingface_hub import login
import imageio
import trimesh


from trellis2.utils.vae_helpers import *
from trellis2.utils import render_utils


# Helper for Python < 3.10 compatibility
class nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


# ---------------------------------------------------------------------
# Environment tweaks
# ---------------------------------------------------------------------
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["ATTN_BACKEND"] = "xformers"

# Check if we are in a notebook
try:
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.modules.sparse import SparseTensor
    import trellis2.models as models
except ImportError:
    print(
        "Error: trellis2 module not found. Make sure you are in the correct environment."
    )

from PIL import Image

os.environ["PYOPENGL_PLATFORM"] = (
    "egl"  # must come before importing pyrender, OpenGL, etc.
)

# --- pyrender imports must come after pyglet option set ---
import pyglet

pyglet.options["shadow_window"] = False


def generate_combinations_v1(
    ss_dir: Path, shape_dir: Path
) -> Iterator[Tuple[str, Path, Path]]:
    ss_map: Dict[str, Path] = {p.stem: p for p in ss_dir.glob("*.npz")}
    shape_map: Dict[str, Path] = {p.stem: p for p in shape_dir.glob("*.npz")}

    with open("datasets/AutoBrep_Dataset/vae_exps.json") as f:
        d = json.load(f)
        for item in d:
            name = item[0]
            ss_map_name = item[1]
            shape_map_name = item[2]
            if ss_map_name in ss_map and shape_map_name in shape_map:
                yield name, ss_map[ss_map_name], shape_map[shape_map_name], ss_map[
                    shape_map_name
                ], shape_map[ss_map_name]


def generate_combinations_v2(
    ss_dir: Path, shape_dir: Path
) -> Iterator[Tuple[str, Path, Path]]:
    ss_map: Dict[str, Path] = {p.stem: p for p in ss_dir.glob("*.npz")}
    shape_map: Dict[str, Path] = {p.stem: p for p in shape_dir.glob("*.npz")}

    with open("o-voxel/examples/parquet_name2sha.json", "r") as file:
        parquet_name2sha = json.load(file)

    for key in parquet_name2sha.keys():
        if parquet_name2sha[key] in shape_map:
            yield key, shape_map[parquet_name2sha[key]], (
                ss_map[parquet_name2sha[key]]
                if parquet_name2sha[key] in ss_map
                else None
            )


def generate_combinations_v3(
    ss_dir: Path, shape_dir: Path
) -> Iterator[Tuple[str, Path, Path]]:
    ss_map: Dict[str, Path] = {p.stem: p for p in ss_dir.glob("*.npz")}
    shape_map: Dict[str, Path] = {p.stem: p for p in shape_dir.glob("*.npz")}

    for key in shape_map.keys():
        yield key, shape_map[key], (ss_map[key] if key in ss_map else None)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

"""
python VAEdecoding.py  --dataset_root datasets/AutoBrep_Dataset --out_dir datasets/AutoBrep_Dataset/decode_shapes/exp1  --low_vram --use_ss_decoder --ss_target_res 64 --pool_on_cpu --decode_resolutions 1024 512 256 --decimate_faces 16777216 --dtype fp32
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    # parser.add_argument("hf_token", type=str)
    parser.add_argument(
        "--dataset_root", type=Path, default=Path("datasets/AutoBrep_Dataset")
    )
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--model_id", type=str, default="microsoft/TRELLIS.2-4B")
    parser.add_argument(
        "--decoder_pretrained",
        type=list[str],
        default=[
            "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16",
            "microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16",
        ],
    )
    parser.add_argument(
        "--low_vram", action="store_true", help="Enable TRELLIS.2 low_vram mode."
    )
    parser.add_argument(
        "--use_ss_decoder",
        action="store_true",
        help="Use sparse_structure_decoder to compute coords.",
    )
    parser.add_argument(
        "--ss_target_res",
        type=int,
        default=64,
        help="Target resolution for SS decoder coords.",
    )
    parser.add_argument(
        "--pool_on_cpu",
        action="store_true",
        help="Pool SS occupancy on CPU to save VRAM.",
    )
    parser.add_argument(
        "--decode_resolutions", type=int, nargs="+", default=[1024, 512, 256]
    )
    parser.add_argument(
        "--decimate_faces",
        type=int,
        default=16777216,
        help="If >0, simplify mesh to this face count.",
    )
    parser.add_argument(
        "--dtype", type=str, default="fp32", choices=["fp16", "bf16", "fp32"]
    )
    args = parser.parse_args()

    # login(token=args.hf_token)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This script expects a CUDA GPU.")

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    compute_dtype = dtype_map[args.dtype]

    LOGGER.info(f"Loading pipeline {args.model_id} ...")

    # Load Pipeline
    decoder_slat = models.from_pretrained(args.decoder_pretrained[0]).eval().cuda()
    decoder_slat.low_vram = bool(args.low_vram)

    decoder_shape = models.from_pretrained(args.decoder_pretrained[1]).eval().cuda()
    decoder_shape.low_vram = bool(args.low_vram)

    torch.cuda.reset_peak_memory_stats()
    log_cuda_memory("after_pipeline_load")

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Create context for mixed precision
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=compute_dtype)
        if compute_dtype != torch.float32
        else nullcontext()
    )

    # Run Inference
    with torch.inference_mode(), autocast_ctx:
        for res in [1024, 512, 256]:
            # Construct paths based on resolution
            ss_dir = (
                args.dataset_root / "ss_latents" / f"ss_enc_conv3d_16l8_fp16_64_{res}"
            )
            shape_dir = (
                args.dataset_root
                / "shape_latents"
                / f"shape_enc_next_dc_f16c32_fp16_{res}"
            )

            if not ss_dir.exists() or not shape_dir.exists():
                LOGGER.warning(
                    f"Missing dirs for res={res}: \n  {ss_dir} \n  {shape_dir}"
                )
                continue

            LOGGER.info(
                f"Decoding resolution={res} from:\n  ss: {ss_dir}\n  shape: {shape_dir}"
            )

            # Process files
            exp_combinations = generate_combinations_v3(ss_dir, shape_dir)

            for (
                key,
                shape_path,
                ss_path,
            ) in exp_combinations:
                torch.cuda.reset_peak_memory_stats()

                LOGGER.info(f"Processing: {key}")

                # 1. Load latents (CPU)
                feats_np, coords_np = load_shape_latent_npz(shape_path)
                z_np = load_ss_latent_npz(ss_path) if args.use_ss_decoder else None

                # 2. Move to GPU
                feats = torch.from_numpy(feats_np).to(
                    device=device, dtype=compute_dtype, non_blocking=True
                )

                enc_coords = torch.from_numpy(coords_np).to(
                    device=device, dtype=torch.int32, non_blocking=True
                )
                enc_coords = add_batch_dim_if_missing(enc_coords)

                LOGGER.info(
                    f"Loaded latents: feats shape {feats.shape}, coords shape {enc_coords.shape}"
                )

                # 3. Optional: Decode Sparse Structure coords
                if args.use_ss_decoder:
                    LOGGER.info("Decoding sparse structure coordinates...")
                    z_s = torch.from_numpy(z_np).to(
                        device=device, dtype=compute_dtype, non_blocking=True
                    )
                    LOGGER.info(f"Loaded latents: z_s shape {z_s.shape} ")
                    if z_s.ndim == 4:
                        z_s = z_s.unsqueeze(0)

                    dec_coords, decoded = decode_sparse_structure_coords(
                        decoder_slat,
                        z_s,
                        target_resolution=64,
                        pool_on_cpu=args.pool_on_cpu,
                    )
                    dec_coords = dec_coords.to(device)

                    LOGGER.info(f"Decoded latents coords shape {dec_coords.shape}")
                    coords = dec_coords
                else:
                    coords = enc_coords

                ## take intersection of dec_coords and enc_coords
                # dec_coords_set = set(map(tuple, dec_coords.cpu().numpy()))
                # enc_coords_set = set(map(tuple, enc_coords.cpu().numpy()))
                # coords_torch = (
                #     torch.from_numpy(np.array(list(dec_coords_set & enc_coords_set)))
                #     .to(torch.int32)
                #     .contiguous()
                #     .to(device)
                # )
                ## update feats to match coords
                # coord_to_index = {tuple(c.cpu().numpy()): i for i, c in enumerate(dec_coords)}
                # indices = [coord_to_index[tuple(c.cpu().numpy())] for c in coords_torch]
                # feats_temp = feats_temp[indices]
                # feats_temp =  feats[indices]  # feats_temp

                # for k in [10, 50, 100, 250, 500, 1500, 3000]:
                # nn_idx, nn_dist = nearest_neighbor_kdtree(coords_torch, coords_torch, k=k)
                # coords = coords_torch[nn_idx[:1]][0]
                # feats = feats_temp[nn_idx[:1]][0]

                # ## NN features
                # alpha = 0.0
                # nn_idx, nn_dist = nearest_neighbor_kdtree(dec_coords, enc_coords)
                # feats = feats[nn_idx] * alpha + feats_temp * (1.0 - alpha)
                # coords = dec_coords

                # LOGGER.info(
                #     f"After intersection, feats shape: {feats.shape}, coords shape: {coords.shape}"
                # )

                # export_voxels_as_cubes_mesh(
                #     args.out_dir
                #     / f"VOXELS_{key}_{res}{'_from_dec-coords.ply' if args.use_ss_decoder else '_from_enc-coords.ply'}",
                #     coords,
                #     voxel_size=1.0,
                # )

                out_name = f"{key}_{res}" + (
                    "_from_dec-coords" if args.use_ss_decoder else "_from_enc-coords"
                )
                out_path = args.out_dir / f"{out_name}"

                # 4. Construct SparseTensor
                shape_slat = SparseTensor(feats=feats, coords=coords)
                LOGGER.info(
                    f"Constructing SparseTensor with feats {feats.shape} and coords {coords.shape}"
                )

                log_cuda_memory(f"{key}: before_decode")

                # 5. Decode Meshes
                meshes = decode_meshes_from_shape_slat(
                    decoder_shape.cuda(),
                    shape_slat,
                    resolution=res,
                    return_subs=False,
                )

                # 6. Post-processing
                mesh = meshes[0]
                mesh.fill_holes()

                if args.decimate_faces and args.decimate_faces > 0:
                    mesh.simplify(args.decimate_faces)

                # 7. Export
                v = mesh.vertices.detach().cpu().numpy().astype(np.float32)
                f = mesh.faces.detach().cpu().numpy().astype(np.int32)

                # write_ply_binary(f"{out_path}.ply", v, f)

                trellis_video = render_utils.make_vis_frames(
                    render_utils.render_video(mesh, num_frames=70)
                )
                imageio.mimsave(
                    f"{out_path}.mp4",
                    trellis_video,
                    fps=15,
                )

                img = render_voxels_pyvista(
                    coords_np,
                    surf_idx=np.arange(len(coords_np)),
                    bound_idx=None,
                    out_path=f"{out_path}.jpg",
                )

                log_cuda_memory(f"{key}: after_export")
                LOGGER.info(f"Wrote {out_path}")

                # 8. Cleanup to prevent VRAM accumulation
                cleanup_cuda(shape_slat, feats, coords, meshes, mesh)
                if args.use_ss_decoder:
                    cleanup_cuda(z_s)

    LOGGER.info("Done.")


# Python <3.10 compatibility helper
class nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
