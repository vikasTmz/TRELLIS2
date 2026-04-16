#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


import torch
import json

# from trellis2.pipelines import Trellis2ImageTo3DPipeline
# from trellis2.modules.sparse import SparseTensor

from trellis2.utils.vae_helpers import *


def generate_combinations(
    ss_dir: Path, shape_dir: Path
) -> Iterator[Tuple[str, Path, Path]]:
    ss_map: Dict[str, Path] = {p.stem: p for p in ss_dir.glob("*.npz")}
    shape_map: Dict[str, Path] = {p.stem: p for p in shape_dir.glob("*.npz")}

    with open("o-voxel/examples/parquet_name2sha.json", "r") as file:
        parquet_name2sha = json.load(file)

    for i in range(100):
        surface_name = f"sample_{i:06d}_surface"
        edge_name = f"sample_{i:06d}_edges"

        if (
            parquet_name2sha[surface_name] in shape_map
            and parquet_name2sha[edge_name] in shape_map
        ):
            yield f"sample_{i:06d}", shape_map[
                parquet_name2sha[surface_name]
            ], shape_map[parquet_name2sha[edge_name]]


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

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This script expects a CUDA GPU.")

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    compute_dtype = dtype_map[args.dtype]

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
            exp_combinations = generate_combinations(ss_dir, shape_dir)

            for (
                key,
                surface_path,
                boundary_path,
            ) in exp_combinations:
                torch.cuda.reset_peak_memory_stats()

                LOGGER.info(f"Processing: {key}")

                # 1. Load latents (CPU)
                _, surf_coords_np = load_shape_latent_npz(surface_path)
                _, bound_coords_np = load_shape_latent_npz(boundary_path)

                LOGGER.info(
                    f"Loaded latents:  surf_enc_coords = {surf_coords_np.shape}, bound_enc_coords = {bound_coords_np.shape}"
                )

                surf_coords_np = surf_coords_np[:, 1:4]
                bound_coords_np = bound_coords_np[:, 1:4]

                # take intersection of surf_coords_np and bound_coords_np

                surf_union_bound_coords = np.array(
                    list(
                        set(map(tuple, surf_coords_np))
                        & set(map(tuple, bound_coords_np))
                    )
                )

                surf_minus_bound_coords = np.array(
                    list(
                        set(map(tuple, surf_coords_np))
                        - set(map(tuple, surf_union_bound_coords))
                    )
                )

                surf_and_bound_coords_1 = (
                    np.concatenate(
                        [
                            surf_minus_bound_coords,
                            bound_coords_np,
                        ],
                        axis=0,
                    )
                    if len(surf_minus_bound_coords) > 0
                    else bound_coords_np
                )

                surf_and_bound_coords_2 = (
                    np.concatenate(
                        [
                            surf_minus_bound_coords,
                            surf_union_bound_coords,
                        ],
                        axis=0,
                    )
                    if len(surf_minus_bound_coords) > 0
                    else surf_union_bound_coords
                )

                out_name = f"{key}_{res}" + (
                    "_from_dec-coords" if args.use_ss_decoder else "_from_enc-coords"
                )
                toCombine = True
                images = []
                out_path = None
                for coords, suffix, surf_idx, bound_idx in [
                    [
                        surf_coords_np,
                        "surf",
                        np.arange(len(surf_coords_np)),
                        None,
                    ],
                    [bound_coords_np, "bound", None, np.arange(len(bound_coords_np))],
                    [
                        surf_and_bound_coords_1,
                        "comb",
                        np.arange(len(surf_minus_bound_coords)),
                        len(surf_minus_bound_coords) + np.arange(len(bound_coords_np)),
                    ],
                    [
                        surf_union_bound_coords,
                        "bound_union",
                        None,
                        np.arange(len(surf_union_bound_coords)),
                    ],
                    [
                        surf_and_bound_coords_2,
                        "comb",
                        np.arange(len(surf_minus_bound_coords)),
                        len(surf_minus_bound_coords)
                        + np.arange(len(surf_union_bound_coords)),
                    ],
                ]:
                    if not toCombine:
                        out_path = args.out_dir / f"{out_name}_{suffix}.jpg"
                    img = render_voxels_pyvista(
                        coords,
                        surf_idx=surf_idx,
                        bound_idx=bound_idx,
                        out_path=out_path,
                    )
                    if toCombine:
                        images.append(img)

                if toCombine:
                    gap = 20
                    total_width = sum(img.width for img in images) + gap * (
                        len(images) - 1
                    )
                    max_height = max(img.height for img in images)
                    combined = Image.new("RGB", (total_width, max_height), "white")
                    x = 0
                    for img in images:
                        combined.paste(img, (x, 0))
                        x += img.width + gap
                    combined.save(args.out_dir / f"{out_name}.jpg")

                LOGGER.info(f"Wrote {out_name}")

    LOGGER.info("Done.")


# Python <3.10 compatibility helper
class nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
