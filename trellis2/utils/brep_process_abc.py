import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import trimesh

from trellis2.utils.brep_helpers import *
from trellis2.datasets.abc_data import *

PathLike = Union[str, Path]
RowDict = Dict[str, Any]


def export_full_model_glb(
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
    surface_obj_path: PathLike,
) -> None:
    print(f"[PID {os.getpid()}] Generating full model GLB: {surface_obj_path}")

    mesh = trimesh.Trimesh(
        vertices=surface_vertices,
        faces=surface_faces,
        process=False,
    )
    scene = trimesh.Scene(mesh)
    scene.export(surface_obj_path)

    print(f"[PID {os.getpid()}] Saved surface: {surface_obj_path}")


def export_boundaries_glb(
    edge_points: np.ndarray,
    edge_segments: np.ndarray,
    edge_obj_path: PathLike,
    pts_obj: Optional[PathLike] = None,
) -> None:
    print(f"[PID {os.getpid()}] Calculating boundary edges: {edge_obj_path}")

    boundary_segments = np.concatenate(
        [
            edge_points[edge_segments[:, 0:1]],
            edge_points[edge_segments[:, 1:2]],
        ],
        axis=1,
    )

    points = sample_and_export_boundary_points(
        boundary_segments,
        output_path=pts_obj,
        points_per_unit=2e3,
        min_dist=5e-5,
        shuffle_filter=True,
        seed=0,
    )

    mesh = points_to_metaball_mesh_new(
        points,
        radius=1e-100,
        smooth_k=1e3,
        voxel_size=2.5e-3,
        padding=4e-1,
        eps=1e-6,
    )

    export_mesh(mesh, None, edge_obj_path)

    print(f"[PID {os.getpid()}] Saved edges: {edge_obj_path}")


def export_gvd(
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
    edge_points: np.ndarray,
    edge_segments: np.ndarray,
    edge_obj_path: PathLike,
    pts_obj: Optional[PathLike] = None,
) -> None:
    print(f"[PID {os.getpid()}] Calculating boundary edges: {edge_obj_path}")

    boundary_segments = np.concatenate(
        [
            edge_points[edge_segments[:, 0:1]],
            edge_points[edge_segments[:, 1:2]],
        ],
        axis=1,
    )

    # Convert group dictionary into flat mesh arrays plus one B-Rep patch id per triangle.
    V, F, face_patch_ids = arrays_from_importer_groups(surface_vertices, groups)

    # Optional: infer topological patch adjacency from shared/quantized geometric edges.
    adjacent_patch_pairs, inferred_boundary_segments = patch_adjacency_from_mesh(
        V,
        F,
        face_patch_ids,
        ndigits=6,
    )

    # Full GVD: includes sheets between any two patches that become nearest neighbors
    # inside the sampled bounding box.
    # gvd = compute_generalized_voronoi(
    #     V,
    #     F,
    #     face_patch_ids,
    #     resolution=256,
    #     padding=0.08,
    #     max_distance=None,
    #     inside_only=False,
    #     allowed_pairs=None,
    #     neighborhood=26,
    #     batch_size=200_000,
    #     verbose=True,
    # )
    gvd = compute_generalized_voronoi(
        V,
        F,
        face_patch_ids,
        resolution=160,
        padding=0.08,
        allowed_pairs=adjacent_patch_pairs,
        neighborhood=26,
        verbose=True,
        inside_only=True,
    )

    paths = gvd.export(output_path + "_boundary")
    print(paths)

    # Optional: export your detected B-Rep edges as OBJ lines.
    export_segments_obj(boundary_segments, output_path + "_boundary.obj")


def process_abc_to_trellismesh(
    geom: Dict[str, np.ndarray],
    surface_obj_path: PathLike,
    edge_obj_path: PathLike,
    pts_obj: Optional[PathLike] = None,
) -> None:
    export_full_model_glb(
        geom["surface_vertices"],
        geom["surface_faces"],
        surface_obj_path,
    )
    # export_boundaries_glb(
    #     geom["edge_points"],
    #     geom["edge_segments"],
    #     edge_obj_path,
    #     pts_obj,
    # )
    export_gvd(
        geom["surface_vertices"],
        geom["surface_faces"],
        geom["edge_points"],
        geom["edge_segments"],
        edge_obj_path,
        pts_obj,
    )


def process_one_sample(i: int, row: RowDict, out_dir: PathLike) -> int:
    # Move the expensive step into the worker
    geom = extract_surface_and_edge_geometry(row, weld_tolerance=1e-6)

    surface_obj = Path(out_dir) / f"sample_{i:06d}_surface.glb"
    edge_obj = Path(out_dir) / f"sample_{i:06d}_edges.glb"
    pts_obj = Path(out_dir) / f"sample_{i:06d}_edgepoints.ply"

    process_abc_to_trellismesh(
        geom,
        surface_obj,
        edge_obj,
        # pts_obj,
    )
    return i


if __name__ == "__main__":
    parquet_dir = "/home/vthamizharas/Documents/TRELLIS.2/datasets/AutoBrep_Dataset/"
    out_dir = Path(
        "/home/vthamizharas/Documents/TRELLIS.2/datasets/AutoBrep_Dataset/raw/brepgen-3ef62ce8a7854009b733e55be707626c-000002-000000-0/"
    )
    out_dir.mkdir(exist_ok=True)

    dataset = ParquetRowIterable(
        paths=parquet_dir,
        columns=GEOMETRY_COLUMNS,
        deserialize_fn=deserialize_geometry_row,
        post_filter=validate_geometry_row,
        map_func=None,  # important: do not extract geometry here
        limit=100,
    )

    for i, row in enumerate(dataset):
        process_one_sample(i, row, out_dir)
        print(f"Finished sample {i:06d}")

    # max_sample_workers = min(1, os.cpu_count() or 1)

    # futures = []
    # with ThreadPoolExecutor(max_workers=max_sample_workers) as executor:
    #     for i, row in enumerate(dataset):
    #         futures.append(executor.submit(process_one_sample, i, row, out_dir))

    #     for future in as_completed(futures):
    #         i = future.result()
    #         print(f"Finished sample {i:06d}")

    print("All samples processed.")
