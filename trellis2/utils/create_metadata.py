import hashlib


def compute_file_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read the file in 4KB chunks
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


import glob

files = glob.glob(
    "/home/vthamizharas/Documents/TRELLIS.2/datasets/Thingi10K/gt_shapes/*glb"
)

with open(
    "/home/vthamizharas/Documents/TRELLIS.2/datasets/Thingi10K/metadata.csv",
    "w",
) as csv_file:
    csv_file.write(f"sha256,local_path\n")
    for filename in files:
        sha256 = compute_file_sha256(filename)
        # write to CSV
        filename = filename.split(
            "/home/vthamizharas/Documents/TRELLIS.2/datasets/Thingi10K/"
        )[-1]
        csv_file.write(f"{sha256},{filename}\n")
