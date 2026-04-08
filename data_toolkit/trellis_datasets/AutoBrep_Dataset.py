import os
import argparse
from concurrent.futures import ThreadPoolExecutor
import traceback
from tqdm import tqdm
import pandas as pd
import objaverse.xl as oxl
from utils import get_file_hash


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--source",
        type=str,
        default="sketchfab",
        help="Data source to download annotations from (github, sketchfab)",
    )


def get_metadata(source, **kwargs):
    pass


def download(metadata, output_dir, **kwargs):
    pass


def foreach_instance(
    metadata, output_dir, func, max_workers=None, desc="Processing objects"
) -> pd.DataFrame:
    import os
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    import tempfile
    import zipfile

    # load metadata
    metadata = metadata.to_dict("records")

    # processing objects
    records = []
    max_workers = max_workers or os.cpu_count()
    try:
        print(f"Using {max_workers} workers for processing.")

        # with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(
        #     total=len(metadata), desc=desc
        # ) as pbar:

        #     def worker(metadatum):
        #         try:
        #             local_path = metadatum["local_path"]
        #             sha256 = metadatum["sha256"]

        #             file = os.path.join(output_dir, local_path)
        #             record = func(file, metadatum)
        #             if record is not None:
        #                 records.append(record)
        #             pbar.update()
        #         except Exception as e:
        #             print(f"Error processing object {sha256}: {e}")
        #             pbar.update()

        with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(
            total=len(metadata), desc=desc
        ) as pbar:

            def worker(metadatum):
                try:
                    if "local_path" in metadatum:
                        local_path = metadatum["local_path"]
                        file = os.path.join(output_dir, local_path)
                    else:
                        local_path = None
                        file = None

                    sha256 = metadatum["sha256"]

                    record = func(file, metadatum=metadatum)
                    if record is not None:
                        records.append(record)
                    pbar.update()
                except Exception as e:
                    print(f"Error processing object {sha256}: {e}")
                    traceback.print_exc()
                    pbar.update()

            executor.map(worker, metadata)
            executor.shutdown(wait=True)
    except:
        print("Error happened during processing.")

    return pd.DataFrame.from_records(records)
