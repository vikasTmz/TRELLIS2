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

torch.set_grad_enabled(False)


def is_valid_sparse_tensor(tensor):
    return torch.isfinite(tensor.feats).all() and torch.isfinite(tensor.coords).all()


def clear_cuda_error():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def to_sparse_coords(tensor, threshold=0.0):
    """
    Converts a dense voxel grid (logits or probabilities) to sparse coordinates.
    Assumes tensor shape (1, D, H, W) or (D, H, W).
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    if tensor.dim() == 5:
        tensor = tensor.squeeze(0).squeeze(0)

    # Assuming output is logits, use 0.0 threshold. If prob, use 0.5.
    # Adjust based on specific model output activation.
    mask = tensor > threshold
    coords = torch.nonzero(mask)
    return coords.cpu().numpy().astype(np.int16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Directory containing the metadata and data",
    )
    parser.add_argument(
        "--shape_latent_root",
        type=str,
        default=None,
        help="Directory to save the shape latent files",
    )
    parser.add_argument(
        "--ss_latent_root",
        type=str,
        default=None,
        help="Directory to load the ss latent files",
    )
    parser.add_argument(
        "--resolution", type=int, default=1024, help="Sparse voxel resolution"
    )
    parser.add_argument(
        "--shape_latent_name",
        type=str,
        default=None,
        help="Name of the shape latent files",
    )
    # Changed: Default to a decoder path
    parser.add_argument(
        "--dec_pretrained",
        type=str,
        default="microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16",
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
    opt.shape_latent_root = opt.shape_latent_root or opt.root
    opt.ss_latent_root = opt.ss_latent_root or opt.root

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
        decoder = getattr(models, cfg.models.decoder.name)(
            **cfg.models.decoder.args
        ).cuda()

        # Changed: Loading decoder checkpoint
        ckpt_path = os.path.join(
            opt.model_root, opt.dec_model, "ckpts", f"decoder_{opt.ckpt}.pt"
        )
        decoder.load_state_dict(torch.load(ckpt_path), strict=False)
        decoder.eval()
        print(f"Loaded decoder from {ckpt_path}")

    os.makedirs(
        os.path.join(opt.shape_latent_root, "ss_latents", latent_name, "new_records"),
        exist_ok=True,
    )

    # --- Metadata Loading ---
    if not os.path.exists(os.path.join(opt.root, "metadata.csv")):
        raise ValueError("metadata.csv not found")

    metadata = pd.read_csv(os.path.join(opt.root, "metadata.csv")).set_index("sha256")

    if os.path.exists(os.path.join(opt.root, "aesthetic_scores", "metadata.csv")):
        metadata = metadata.combine_first(
            pd.read_csv(
                os.path.join(opt.root, "aesthetic_scores", "metadata.csv")
            ).set_index("sha256")
        )

    # Merge existing latent records to ensure we only process objects that exist
    if os.path.exists(
        os.path.join(
            opt.ss_latent_root, "ss_latents", latent_name, "new_records/part_0.csv"
        )
    ):
        metadata = metadata.combine_first(
            pd.read_csv(
                os.path.join(
                    opt.ss_latent_root,
                    "ss_latents",
                    latent_name,
                    "new_records/part_0.csv",
                )
            ).set_index("sha256")
        )

    metadata = metadata.reset_index()

    # Changed Logic: We need items that ARE encoded, but NOT yet reconstructed
    if "ss_latent_encoded" in metadata.columns:
        metadata = metadata[metadata["ss_latent_encoded"] == True]
    else:
        print(
            "Warning: 'ss_latent_encoded' column not found. Assuming check by file existence."
        )

    if opt.instances is None:
        if "ss_latent_decoded" in metadata.columns:
            metadata = metadata[metadata["ss_latent_decoded"] != True]
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

    # filter out objects that are already processed (reconstructed)
    save_dir = os.path.join(opt.shape_latent_root, "ss_latents", latent_name)
    os.makedirs(save_dir, exist_ok=True)

    sha256_list = os.listdir(save_dir)
    sha256_list = [os.path.splitext(f)[0] for f in sha256_list if f.endswith(".npz")]
    for sha256 in sha256_list:
        records.append({"sha256": sha256, "ss_latent_decoded": True})
    print(f"Found {len(sha256_list)} already reconstructed objects")
    metadata = metadata[~metadata["sha256"].isin(sha256_list)]

    print(f"Processing {len(metadata)} objects for reconstruction...")

    sha256s = list(metadata["sha256"].values)
    load_queue = Queue(maxsize=32)
    with ThreadPoolExecutor(max_workers=32) as loader_executor, ThreadPoolExecutor(
        max_workers=32
    ) as saver_executor:

        # --- Data Loader (Loads Latents z) ---
        def loader(sha256):
            try:
                # Load the latent vector Z
                latent_path = os.path.join(
                    opt.ss_latent_root,
                    "ss_latents",
                    opt.shape_latent_name,
                    f"{sha256}.npz",
                )
                data = np.load(latent_path)
                z = torch.from_numpy(data["z"])
                load_queue.put((sha256, z))
            except Exception as e:
                print(f"[Loader Error] {sha256}: {e}")
                load_queue.put((sha256, None))

        loader_executor.map(loader, sha256s)

        # --- Data Saver (Saves Reconstructed Coords) ---
        def saver(sha256, pack):
            save_path = os.path.join(save_dir, f"{sha256}.npz")
            np.savez_compressed(save_path, **pack)
            records.append(
                {
                    "sha256": sha256,
                    "ss_latent_decoded": True,
                    "shape_latent_tokens": pack["coords"].shape[0],
                }
            )

        # --- Main Decoding Loop ---
        for _ in tqdm(range(len(sha256s)), desc="Decoding latents"):
            try:
                sha256, z = load_queue.get()
                if z is None:
                    print(f"[Skip] {sha256}: Failed to load latent input")
                    continue

                # Prepare input: (B, C, D, H, W) or (B, C) depending on architecture
                # The encoder outputted z[0].cpu().numpy(), so we unsqueeze to add batch dim back
                z = z.cuda().unsqueeze(0).float()

                # Run Decoder
                # Some decoders return raw logits, some return probabilities
                out = decoder(z)
                torch.cuda.synchronize()

                if not torch.isfinite(out).all():
                    print(f"[Skip] {sha256}: Non-finite reconstruction")
                    clear_cuda_error()
                    continue

                # Convert dense output back to sparse coords for efficient storage
                # Assumes output is (1, 1, D, H, W) or similar
                coords = to_sparse_coords(
                    out, threshold=0.0
                )  # Threshold 0.0 for logits

                pack = {"coords": coords}
                saver_executor.submit(saver, sha256, pack)

            except Exception as e:
                print(f"[Error] {sha256}: {e}")
                # print(e.with_traceback())
                clear_cuda_error()
                continue

        saver_executor.shutdown(wait=True)

    # Save records
    records = pd.DataFrame.from_records(records)
    records.to_csv(
        os.path.join(
            opt.shape_latent_root,
            "ss_latents",
            latent_name,
            "new_records",
            f"part_{opt.rank}.csv",
        ),
        index=False,
    )
