import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import json
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import trellis2.models as models
import trellis2.modules.sparse as sp

torch.set_grad_enabled(False)


def clear_cuda_error():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str, required=True, help="Directory containing metadata"
    )
    parser.add_argument(
        "--dual_grid_root",
        type=str,
        default=None,
        help="Directory to save the reconstructed dual grids (output)",
    )
    parser.add_argument(
        "--shape_latent_root",
        type=str,
        default=None,
        help="Directory containing the shape latent files (input)",
    )
    parser.add_argument(
        "--resolution", type=int, default=1024, help="Sparse voxel resolution"
    )
    # Changed: Default to a decoder path
    parser.add_argument(
        "--dec_pretrained",
        type=str,
        default="microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16",
        help="Pretrained decoder model",
    )
    parser.add_argument("--model_root", type=str, help="Root directory of models")
    parser.add_argument(
        "--dec_model",
        type=str,
        help="Decoder model. If specified, use this model instead of pretrained model",
    )
    parser.add_argument("--ckpt", type=str, help="Checkpoint to load")
    parser.add_argument(
        "--instances", type=str, default=None, help="Instances to process"
    )
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)

    opt = parser.parse_args()
    opt = edict(vars(opt))
    opt.dual_grid_root = opt.dual_grid_root or opt.root
    opt.shape_latent_root = opt.shape_latent_root or opt.root

    # --- Load Decoder Model ---
    if opt.dec_model is None:
        latent_name = f'{opt.dec_pretrained.split("/")[-1]}_{opt.resolution}'
        decoder = models.from_pretrained(opt.dec_pretrained).eval().cuda()
    else:
        latent_name = f'{opt.dec_model.split("/")[-1]}_{opt.ckpt}_{opt.resolution}'
        cfg = edict(
            json.load(
                open(os.path.join(opt.model_root, opt.dec_model, "config.json"), "r")
            )
        )
        # Using cfg.models.decoder
        decoder = getattr(models, cfg.models.decoder.name)(
            **cfg.models.decoder.args
        ).cuda()

        ckpt_path = os.path.join(
            opt.model_root, opt.dec_model, "ckpts", f"decoder_{opt.ckpt}.pt"
        )
        decoder.load_state_dict(torch.load(ckpt_path), strict=False)
        decoder.eval()
        print(f"Loaded decoder from {ckpt_path}")

    # Output directory for reconstructions
    os.makedirs(
        os.path.join(
            opt.dual_grid_root,
            "shape_latents",
            latent_name,
            f"dual_grid_{opt.resolution}_recon",
            "new_records",
        ),
        exist_ok=True,
    )

    # --- Metadata Loading ---
    if not os.path.exists(os.path.join(opt.root, "metadata.csv")):
        raise ValueError("metadata.csv not found")

    metadata = pd.read_csv(os.path.join(opt.root, "metadata.csv")).set_index("sha256")

    # Check for existing Shape Latents (Our Input)
    if os.path.exists(
        os.path.join(
            opt.shape_latent_root, "shape_latents", latent_name, "metadata.csv"
        )
    ):
        metadata = metadata.combine_first(
            pd.read_csv(
                os.path.join(
                    opt.shape_latent_root, "shape_latents", latent_name, "metadata.csv"
                )
            ).set_index("sha256")
        )

    metadata = metadata.reset_index()

    # Filter: We need items that HAVE shape latents, but NOT yet reconstructed geometry
    if "shape_latent_encoded" in metadata.columns:
        metadata = metadata[metadata["shape_latent_encoded"] == True]

    if opt.instances is None:
        if "dual_grid_reconstructed" in metadata.columns:
            metadata = metadata[metadata["dual_grid_reconstructed"] != True]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, "r") as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(",")
        metadata = metadata[metadata["sha256"].isin(instances)]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []

    # --- Filter existing output files ---
    save_dir = os.path.join(
        opt.dual_grid_root,
        "shape_latents",
        latent_name,
        f"dual_grid_{opt.resolution}_recon",
    )
    os.makedirs(save_dir, exist_ok=True)

    existing_sha256 = set()
    # Check for .npz since we are saving as npz in this decoder script
    for f in os.listdir(save_dir):
        if f.endswith(".npz"):
            existing_sha256.add(os.path.splitext(f)[0])

    for sha256 in existing_sha256:
        records.append({"sha256": sha256, "dual_grid_reconstructed": True})

    print(f"Found {len(existing_sha256)} processed objects")
    metadata = metadata[~metadata["sha256"].isin(existing_sha256)]

    print(f"Processing {len(metadata)} objects...")

    sha256s = list(metadata["sha256"].values)
    load_queue = Queue(maxsize=32)

    with ThreadPoolExecutor(max_workers=32) as loader_executor, ThreadPoolExecutor(
        max_workers=32
    ) as saver_executor:

        # --- Loader: Reads Shape Latents (.npz) ---
        def loader(sha256):
            try:
                # Load the latent (feats + coords)
                data = np.load(
                    os.path.join(
                        opt.shape_latent_root,
                        "shape_latents",
                        latent_name,
                        f"{sha256}.npz",
                    )
                )

                # feats = torch.from_numpy(data["feats"])
                coords = torch.from_numpy(data["coords"])

                # The encoder saved coords as (N, 3), stripped of batch index.
                # The SparseTensor expects (N, 4) where col 0 is batch index.
                # Since batch size is 1, we prepend 0s.
                batch_idx = torch.zeros((coords.shape[0], 1), dtype=coords.dtype)
                full_coords = torch.cat([batch_idx, coords], dim=1).int()

                # Reconstruct the SparseTensor input for the decoder
                z = sp.SparseTensor(feats, full_coords)

                load_queue.put((sha256, z))
            except Exception as e:
                print(f"[Loader Error] {sha256}: {e}")
                load_queue.put((sha256, None))

        loader_executor.map(loader, sha256s)

        # --- Saver: Writes Reconstructed Geometry ---
        def saver(sha256, pack):
            save_path = os.path.join(save_dir, f"{sha256}.npz")
            np.savez_compressed(save_path, **pack)
            records.append(
                {
                    "sha256": sha256,
                    "dual_grid_reconstructed": True,
                    "num_vertices": pack["coords"].shape[0],
                }
            )

        # --- Main Decoding Loop ---
        for _ in tqdm(range(len(sha256s)), desc="Decoding geometry"):
            try:
                sha256, z = load_queue.get()
                if z is None:
                    print(f"[Skip] {sha256}: Failed to load input")
                    continue

                z = z.cuda()

                # Decode
                # The decoder typically returns predictions for vertices (geometry) and intersections (topology)
                # Structure depends on specific Trellis model, assuming it mirrors encoder input structure:
                # Returns: vertices (SparseTensor), intersected (SparseTensor or Tensor)
                outputs = decoder(z)
                torch.cuda.synchronize()

                # Handle different return types from the decoder
                # Assuming outputs is a tuple/list: (vertices_tensor, intersected_tensor)
                # or a SparseTensor containing the features.

                if isinstance(outputs, (tuple, list)):
                    pred_vertices = outputs[0]
                    pred_intersected = outputs[1]
                elif hasattr(outputs, "feats"):
                    # If it returns a single SparseTensor, the feats likely contain both info
                    # or need splitting. Assuming tuple for standard Trellis Dual Grid decoder.
                    pred_vertices = outputs
                    pred_intersected = None  # Handle if needed
                else:
                    pred_vertices = outputs
                    pred_intersected = None

                # Validation
                if not torch.isfinite(pred_vertices.feats).all():
                    print(f"[Skip] {sha256}: Non-finite reconstruction")
                    clear_cuda_error()
                    continue

                # Pack for saving
                # We save the raw arrays.
                # To convert to .vxz later, one would use o_voxel.io.write_vxz with these arrays.
                pack = {
                    "coords": pred_vertices.coords[:, 1:]
                    .cpu()
                    .numpy()
                    .astype(np.int16),
                    "vertices": pred_vertices.feats.cpu()
                    .numpy()
                    .astype(np.float16),  # Usually offsets/pos
                }

                if pred_intersected is not None:
                    # If intersected is a SparseTensor
                    if hasattr(pred_intersected, "feats"):
                        pack["intersected"] = (
                            pred_intersected.feats.cpu().numpy() > 0
                        )  # Logits to bool
                    else:
                        pack["intersected"] = pred_intersected.cpu().numpy()

                saver_executor.submit(saver, sha256, pack)

            except Exception as e:
                print(f"[Error] {sha256}: {e}")
                clear_cuda_error()
                continue

        saver_executor.shutdown(wait=True)

    records = pd.DataFrame.from_records(records)
    records.to_csv(
        os.path.join(
            opt.dual_grid_root,
            f"dual_grid_{opt.resolution}_recon",
            latent_name,
            "new_records",
            f"part_{opt.rank}.csv",
        ),
        index=False,
    )
