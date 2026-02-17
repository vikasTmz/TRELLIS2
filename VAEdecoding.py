#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import logging
import os
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import login

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.modules.sparse import SparseTensor

# ---------------------------------------------------------------------
# Environment tweaks (keep if they help your setup)
# ---------------------------------------------------------------------
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["ATTN_BACKEND"] = "xformers"

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
LOGGER = logging.getLogger("decode_shapes")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def log_cuda_memory(tag: str) -> None:
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    LOGGER.info(
        f"[{tag}] CUDA alloc={alloc:.2f}GB reserved={reserved:.2f}GB peak={peak:.2f}GB"
    )


def add_batch_dim_if_missing(coords: torch.Tensor) -> torch.Tensor:
    # Expected SparseTensor coords are often [N,4] = [b,x,y,z]
    if coords.ndim != 2:
        raise ValueError(f"coords must be 2D, got {coords.shape}")
    if coords.shape[1] == 3:
        zeros = torch.zeros(
            (coords.shape[0], 1), dtype=torch.int32, device=coords.device
        )
        coords = torch.cat([zeros, coords.to(torch.int32)], dim=1)
    return coords.to(torch.int32).contiguous()


def load_shape_latent_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        feats = data["feats"]
        coords = data["coords"]
    return feats, coords


def load_ss_latent_npz(path: Path) -> np.ndarray:
    with np.load(path) as data:
        z = data["z"]
    return z


def pair_latent_files(
    ss_dir: Path, shape_dir: Path
) -> Iterator[Tuple[str, Path, Path]]:
    ss_map: Dict[str, Path] = {p.stem: p for p in ss_dir.glob("*.npz")}
    shape_map: Dict[str, Path] = {p.stem: p for p in shape_dir.glob("*.npz")}

    common = sorted(set(ss_map.keys()) & set(shape_map.keys()))
    missing_ss = sorted(set(shape_map.keys()) - set(ss_map.keys()))
    missing_shape = sorted(set(ss_map.keys()) - set(shape_map.keys()))

    if missing_ss:
        LOGGER.warning(
            f"{len(missing_ss)} shape latents have no matching ss latent (example: {missing_ss[0]})"
        )
    if missing_shape:
        LOGGER.warning(
            f"{len(missing_shape)} ss latents have no matching shape latent (example: {missing_shape[0]})"
        )

    for key in common:
        yield key, ss_map[key], shape_map[key]


def decode_sparse_structure_coords(
    pipeline: Trellis2ImageTo3DPipeline,
    z_s: torch.Tensor,
    target_resolution: int = 64,
    pool_on_cpu: bool = True,
) -> torch.Tensor:
    """
    Mirrors TRELLIS.2 approach: decoded = decoder(z_s) > 0, then max_pool3d if needed. :contentReference[oaicite:0]{index=0}
    We optionally do pooling on CPU to avoid creating a huge dense float tensor on GPU.
    """
    decoder = pipeline.models["sparse_structure_decoder"]

    if pipeline.low_vram:
        decoder.to(pipeline.device)

    decoded = decoder(z_s) > 0  # bool tensor

    if pipeline.low_vram:
        decoder.cpu()

    if decoded.shape[2] != target_resolution:
        ratio = decoded.shape[2] // target_resolution

        if pool_on_cpu:
            decoded_cpu = decoded.to("cpu")
            pooled = F.max_pool3d(decoded_cpu.to(torch.float16), ratio, ratio, 0) > 0.5
            decoded = pooled  # stays on CPU
        else:
            # still better to pool in fp16 than fp32
            decoded = F.max_pool3d(decoded.to(torch.float16), ratio, ratio, 0) > 0.5

    # coords = argwhere(decoded)[:, [0,2,3,4]]  :contentReference[oaicite:1]{index=1}
    coords = (
        decoded.nonzero(as_tuple=False)[:, [0, 2, 3, 4]].to(torch.int32).contiguous()
    )
    return coords


def decode_meshes_from_shape_slat(
    pipeline: Trellis2ImageTo3DPipeline,
    shape_slat: SparseTensor,
    resolution: int,
    return_subs: bool = False,
):
    """
    pipeline.decode_shape_slat always calls shape_slat_decoder(..., return_subs=True) :contentReference[oaicite:2]{index=2}
    If you don't need subs, calling the decoder with return_subs=False can reduce memory pressure.
    """
    shape_decoder = pipeline.models["shape_slat_decoder"]
    shape_decoder.set_resolution(resolution)  # :contentReference[oaicite:3]{index=3}

    if pipeline.low_vram:
        shape_decoder.to(pipeline.device)
        shape_decoder.low_vram = True

    ret = shape_decoder(shape_slat, return_subs=return_subs)

    if pipeline.low_vram:
        shape_decoder.cpu()
        shape_decoder.low_vram = False

    if return_subs:
        meshes, subs = ret
        return meshes, subs

    # Be defensive: some implementations may still return a tuple
    if isinstance(ret, tuple) and len(ret) == 2:
        meshes, _subs = ret
        return meshes

    return ret


def write_ply_binary(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """
    Writes a binary_little_endian PLY with vertex positions and triangular faces.
    vertices: (V,3) float32
    faces: (F,3) int32
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    v = np.asarray(vertices, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int32)

    with path.open("wb") as f_out:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {v.shape[0]}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            f"element face {f.shape[0]}\n"
            "property list int int vertex_indices\n"
            "end_header\n"
        )
        f_out.write(header.encode("ascii"))
        f_out.write(v.tobytes())

        counts = np.full((f.shape[0], 1), 3, dtype=np.int32)
        faces_data = np.hstack([counts, f])
        f_out.write(faces_data.tobytes())


def cleanup_cuda(*tensors) -> None:
    for t in tensors:
        try:
            del t
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("hf_token", type=str)
    parser.add_argument(
        "--dataset_root", type=Path, default=Path("datasets/ObjaverseXL_sketchfab")
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

    login(token=args.hf_token)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This script expects a CUDA GPU.")

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    compute_dtype = dtype_map[args.dtype]

    out_dir = args.out_dir or (args.dataset_root / "decode_shapes")
    out_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(f"Loading pipeline {args.model_id} ...")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model_id)
    pipeline.cuda()
    pipeline.low_vram = bool(
        args.low_vram
    )  # low_vram is supported in the pipeline :contentReference[oaicite:4]{index=4}
    LOGGER.info(f"Pipeline loaded. low_vram={pipeline.low_vram}")

    torch.cuda.reset_peak_memory_stats()
    log_cuda_memory("after_pipeline_load")

    # Inference mode is critical for VRAM
    with torch.inference_mode():
        # autocast helps if some ops default to fp32
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=compute_dtype)
            if compute_dtype != torch.float32
            else nullcontext()
        )
        with autocast_ctx:
            for res in args.decode_resolutions:
                ss_dir = (
                    args.dataset_root
                    / "ss_latents"
                    / f"ss_enc_conv3d_16l8_fp16_64_{res}"
                )
                shape_dir = (
                    args.dataset_root
                    / "shape_latents"
                    / f"shape_enc_next_dc_f16c32_fp16_{res}"
                )

                if not ss_dir.exists() or not shape_dir.exists():
                    LOGGER.warning(
                        f"Missing dirs for res={res}: {ss_dir} or {shape_dir}"
                    )
                    continue

                LOGGER.info(
                    f"Decoding resolution={res} from:\n  ss: {ss_dir}\n  shape: {shape_dir}"
                )

                for key, ss_path, shape_path in pair_latent_files(ss_dir, shape_dir):
                    torch.cuda.reset_peak_memory_stats()

                    # ---- Load latents on CPU first
                    feats_np, coords_np = load_shape_latent_npz(shape_path)
                    z_np = load_ss_latent_npz(ss_path) if args.use_ss_decoder else None

                    # ---- Move to GPU with controlled dtype
                    feats = torch.from_numpy(feats_np).to(
                        device=device, dtype=compute_dtype, non_blocking=True
                    )
                    coords = torch.from_numpy(coords_np).to(
                        device=device, dtype=torch.int32, non_blocking=True
                    )
                    coords = add_batch_dim_if_missing(coords)

                    if args.use_ss_decoder:
                        # SS latent to GPU
                        z_s = torch.from_numpy(z_np).to(
                            device=device, dtype=compute_dtype, non_blocking=True
                        )
                        if z_s.ndim == 4:
                            z_s = z_s.unsqueeze(0)

                        coords = decode_sparse_structure_coords(
                            pipeline,
                            z_s,
                            target_resolution=args.ss_target_res,
                            pool_on_cpu=args.pool_on_cpu,
                        ).to(device)

                    shape_slat = SparseTensor(feats=feats, coords=coords)

                    log_cuda_memory(f"{key}: before_decode")

                    # Decode meshes (avoid returning subs unless you need them)
                    meshes = decode_meshes_from_shape_slat(
                        pipeline,
                        shape_slat,
                        resolution=res,
                        return_subs=False,
                    )

                    mesh = meshes[0]
                    mesh.fill_holes()

                    if args.decimate_faces and args.decimate_faces > 0:
                        mesh.simplify(args.decimate_faces)

                    # Move geometry to CPU for export ASAP
                    v = mesh.vertices.detach().cpu().numpy().astype(np.float32)
                    f = mesh.faces.detach().cpu().numpy().astype(np.int32)

                    out_name = f"{key}_{res}" + (
                        "_sscoords" if args.use_ss_decoder else ""
                    )
                    out_path = out_dir / f"{out_name}.ply"
                    write_ply_binary(out_path, v, f)

                    log_cuda_memory(f"{key}: after_export")
                    LOGGER.info(f"Wrote {out_path}")

                    # Free per-item tensors
                    cleanup_cuda(shape_slat, feats, coords, meshes, mesh)

    LOGGER.info("Done.")


# Python <3.10 compatibility helper
class nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
