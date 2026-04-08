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
    "/home/vthamizharas/Documents/TRELLIS.2/datasets/AutoBrep_Dataset/raw/brepgen-3ef62ce8a7854009b733e55be707626c-000002-000000-0/*"
)

with open(
    "/home/vthamizharas/Documents/TRELLIS.2/datasets/AutoBrep_Dataset/raw/metadata.csv",
    "w",
) as csv_file:
    csv_file.write(f"sha256,local_path\n")
    for filename in files:
        sha256 = compute_file_sha256(filename)
        # write to CSV
        filename = filename.split(
            "/home/vthamizharas/Documents/TRELLIS.2/datasets/AutoBrep_Dataset/"
        )[-1]
        csv_file.write(f"{sha256},{filename}\n")
