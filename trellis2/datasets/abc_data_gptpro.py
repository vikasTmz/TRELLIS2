from __future__ import annotations

"""
Utilities for streaming ABC-1M parquet rows and converting each row into:

- a triangulated surface mesh (vertices + triangle faces)
- a polyline edge graph (points + line segments)

This is a simplified replacement for the original tokenization-heavy abc_data.py.
It intentionally keeps the parquet streaming/deserialization path and removes the
CAD tokenization / augmentation logic that is not needed for geometry export.
"""

from pathlib import Path
import random
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
import io

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds
import torch
from torch.utils.data import DataLoader, IterableDataset

try:
    from pytorch_lightning import LightningDataModule
except ImportError:  # pragma: no cover - fallback for non-Lightning environments

    class LightningDataModule:  # type: ignore[override]
        """Lightweight fallback so this module can be used without Lightning."""

        pass


PathLike = Union[str, Path]
RowDict = Dict[str, Any]

GEOMETRY_COLUMNS = [
    "face_points_normalized",
    "edge_points_normalized",
    "face_bbox_world",
    "edge_bbox_world",
    "face_edge_incidence",
]


class ParquetRowIterable(IterableDataset):
    """
    Stream rows from a parquet directory or file via pyarrow.dataset.

    Pipeline per row:
        pre_filter -> deserialize -> post_filter -> map_func -> yield

    Supports pushdown filtering, approximate local shuffling, and optional limits.
    """

    def __init__(
        self,
        paths: PathLike,
        columns: List[str],
        filter_expr: Optional[pc.Expression] = None,
        pre_filter: Optional[Callable[[RowDict], bool]] = None,
        deserialize_fn: Optional[Callable[[RowDict], RowDict]] = None,
        post_filter: Optional[Callable[[RowDict], bool]] = None,
        map_func: Optional[Callable[[RowDict], RowDict]] = None,
        limit: Optional[int] = None,
        shuffle_buffer_size: Optional[int] = None,
        shuffle_seed: int = 9876,
        batch_rows_read: int = 4096,
    ):
        super().__init__()
        self.paths = str(paths)
        self.columns = columns
        self.filter_expr = filter_expr
        self.pre_filter = pre_filter or (lambda row: True)
        self.deserialize_fn = deserialize_fn or (lambda row: row)
        self.post_filter = post_filter or (lambda row: True)
        self.map_func = map_func or (lambda row: row)
        self.limit = limit
        self.shuffle_buffer_size = shuffle_buffer_size
        self.shuffle_seed = shuffle_seed
        self.batch_rows_read = batch_rows_read

    def _iter_rows(self) -> Iterator[RowDict]:
        dataset = ds.dataset(self.paths, format="parquet")
        scanner = dataset.scanner(
            columns=self.columns,
            filter=self.filter_expr,
            batch_size=self.batch_rows_read,
        )

        yielded = 0
        for record_batch in scanner.to_batches():
            columns = {
                name: record_batch.column(i)
                for i, name in enumerate(record_batch.schema.names)
            }
            for row_idx in range(record_batch.num_rows):
                row = {name: columns[name][row_idx].as_py() for name in columns}

                if not self.pre_filter(row):
                    continue

                row = self.deserialize_fn(row)

                if not self.post_filter(row):
                    continue

                row = self.map_func(row)
                yield row

                yielded += 1
                if self.limit is not None and yielded >= self.limit:
                    return

    def __iter__(self) -> Iterator[RowDict]:
        worker_info = torch.utils.data.get_worker_info()
        base_seed = self.shuffle_seed
        if worker_info is not None:
            base_seed = base_seed + worker_info.id * 1000003

        rng = random.Random(base_seed)

        if not self.shuffle_buffer_size or self.shuffle_buffer_size <= 0:
            yield from self._iter_rows()
            return

        buffer: List[RowDict] = []
        for item in self._iter_rows():
            if len(buffer) < self.shuffle_buffer_size:
                buffer.append(item)
                continue

            slot = rng.randrange(len(buffer))
            yield buffer[slot]
            buffer[slot] = item

        rng.shuffle(buffer)
        yield from buffer


def _compute_bbox_center_and_size(
    min_corner: np.ndarray, max_corner: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Match AutoBrep's bbox-to-world conversion convention."""
    center = (min_corner + max_corner) / 2.0
    size = float(np.max(max_corner - min_corner))
    return center, size


def deserialize_array(serialized: bytes) -> np.ndarray:
    """Fallback NumPy blob deserializer used by the ABC parquet files."""
    memfile = io.BytesIO()
    memfile.write(serialized)
    memfile.seek(0)
    return np.load(memfile, allow_pickle=False)


def deserialize_geometry_row(row: RowDict) -> RowDict:
    """Deserialize the parquet byte fields used for geometry reconstruction."""
    return {
        "face_points_normalized": deserialize_array(row["face_points_normalized"]),
        "edge_points_normalized": deserialize_array(row["edge_points_normalized"]),
        "face_bbox_world": deserialize_array(row["face_bbox_world"]),
        "edge_bbox_world": deserialize_array(row["edge_bbox_world"]),
        "face_edge_incidence": deserialize_array(row["face_edge_incidence"]).astype(
            bool
        ),
    }


def validate_geometry_row(row: RowDict) -> bool:
    """Basic shape checks after deserialization."""
    required = set(GEOMETRY_COLUMNS)
    if not required.issubset(row.keys()):
        return False

    face_points = row["face_points_normalized"]
    edge_points = row["edge_points_normalized"]
    face_bbox = row["face_bbox_world"]
    edge_bbox = row["edge_bbox_world"]
    face_edge_incidence = row["face_edge_incidence"]

    if not isinstance(face_points, np.ndarray) or face_points.ndim != 4:
        return False
    if not isinstance(edge_points, np.ndarray) or edge_points.ndim != 3:
        return False
    if (
        not isinstance(face_bbox, np.ndarray)
        or face_bbox.ndim != 2
        or face_bbox.shape[1] != 6
    ):
        return False
    if (
        not isinstance(edge_bbox, np.ndarray)
        or edge_bbox.ndim != 2
        or edge_bbox.shape[1] != 6
    ):
        return False
    if not isinstance(face_edge_incidence, np.ndarray) or face_edge_incidence.ndim != 2:
        return False

    num_faces = face_points.shape[0]
    num_edges = edge_points.shape[0]

    if face_bbox.shape[0] != num_faces:
        return False
    if edge_bbox.shape[0] != num_edges:
        return False
    if face_edge_incidence.shape != (num_faces, num_edges):
        return False

    return True


def convert_normalized_face_points_to_world(
    face_bbox_world: np.ndarray,
    face_points_normalized: np.ndarray,
) -> np.ndarray:
    """
    Convert normalized face UV grids to world coordinates.

    face_bbox_world shape: (num_faces, 6)
    face_points_normalized shape: (num_faces, num_u, num_v, 3)
    return shape: (num_faces, num_u, num_v, 3)
    """
    if face_bbox_world.ndim != 2 or face_bbox_world.shape[1] != 6:
        raise ValueError("face_bbox_world must have shape (num_faces, 6)")
    if face_points_normalized.ndim != 4:
        raise ValueError(
            "face_points_normalized must have shape (num_faces, num_u, num_v, 3)"
        )
    if face_bbox_world.shape[0] != face_points_normalized.shape[0]:
        raise ValueError(
            "Number of face bounding boxes and face point grids must match"
        )

    num_faces, num_u, num_v, _ = face_points_normalized.shape
    world_points: List[np.ndarray] = []

    for face_bbox, face_points in zip(face_bbox_world, face_points_normalized):
        center, size = _compute_bbox_center_and_size(face_bbox[:3], face_bbox[3:])
        world = face_points.reshape(-1, 3) * (size / 2.0) + center
        world_points.append(world.reshape(num_u, num_v, 3))

    return np.stack(world_points, axis=0)


def convert_normalized_edge_points_to_world(
    edge_bbox_world: np.ndarray,
    edge_points_normalized: np.ndarray,
) -> np.ndarray:
    """
    Convert normalized edge point grids to world coordinates.

    edge_bbox_world shape: (num_edges, 6)
    edge_points_normalized shape: (num_edges, num_u, 3)
    return shape: (num_edges, num_u, 3)
    """
    if edge_bbox_world.ndim != 2 or edge_bbox_world.shape[1] != 6:
        raise ValueError("edge_bbox_world must have shape (num_edges, 6)")
    if edge_points_normalized.ndim != 3:
        raise ValueError("edge_points_normalized must have shape (num_edges, num_u, 3)")
    if edge_bbox_world.shape[0] != edge_points_normalized.shape[0]:
        raise ValueError(
            "Number of edge bounding boxes and edge point grids must match"
        )

    world_points: List[np.ndarray] = []
    for edge_bbox, edge_points in zip(edge_bbox_world, edge_points_normalized):
        center, size = _compute_bbox_center_and_size(edge_bbox[:3], edge_bbox[3:])
        world_points.append(edge_points * (size / 2.0) + center)

    return np.stack(world_points, axis=0)


class _PointIndexer:
    """Insert points while optionally welding nearly-identical coordinates."""

    def __init__(self, tolerance: Optional[float] = 1e-6):
        self.tolerance = tolerance
        self.points: List[np.ndarray] = []
        self.lookup: Dict[Tuple[int, int, int], int] = {}

    def _key(self, point: np.ndarray) -> Optional[Tuple[int, int, int]]:
        if self.tolerance is None or self.tolerance <= 0:
            return None
        scaled = np.rint(
            np.asarray(point, dtype=np.float64) / float(self.tolerance)
        ).astype(np.int64)
        return int(scaled[0]), int(scaled[1]), int(scaled[2])

    def add(self, point: np.ndarray) -> int:
        key = self._key(point)
        if key is not None and key in self.lookup:
            return self.lookup[key]

        index = len(self.points)
        self.points.append(np.asarray(point, dtype=np.float32))
        if key is not None:
            self.lookup[key] = index
        return index

    def as_array(self) -> np.ndarray:
        if not self.points:
            return np.zeros((0, 3), dtype=np.float32)
        return np.stack(self.points, axis=0).astype(np.float32)


def _non_degenerate_triangle(
    a: int, b: int, c: int, vertices: np.ndarray, area_eps: float = 1e-12
) -> bool:
    if a == b or b == c or a == c:
        return False
    tri = vertices[[a, b, c]]
    area_twice = np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
    return bool(area_twice > area_eps)


def surface_grids_to_vertices_and_faces(
    face_points_world: np.ndarray,
    weld_tolerance: Optional[float] = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Triangulate per-face point grids into a global triangle mesh.

    Each face grid is triangulated as two triangles per quad in the UV grid.

    Returns:
        vertices: (N, 3) float32
        faces: (M, 3) int64, zero-based indices into vertices
    """
    if face_points_world.ndim != 4 or face_points_world.shape[-1] != 3:
        raise ValueError(
            "face_points_world must have shape (num_faces, num_u, num_v, 3)"
        )

    indexer = _PointIndexer(weld_tolerance)
    faces: List[Tuple[int, int, int]] = []

    for face_grid in face_points_world:
        num_u, num_v, _ = face_grid.shape
        face_vertex_ids = np.empty((num_u, num_v), dtype=np.int64)

        for u in range(num_u):
            for v in range(num_v):
                face_vertex_ids[u, v] = indexer.add(face_grid[u, v])

        for u in range(num_u - 1):
            for v in range(num_v - 1):
                a = int(face_vertex_ids[u, v])
                b = int(face_vertex_ids[u + 1, v])
                c = int(face_vertex_ids[u + 1, v + 1])
                d = int(face_vertex_ids[u, v + 1])

                faces.append((a, b, c))
                faces.append((a, c, d))

    vertices = indexer.as_array()
    filtered_faces = [
        face for face in faces if _non_degenerate_triangle(*face, vertices)
    ]

    if not filtered_faces:
        return vertices, np.zeros((0, 3), dtype=np.int64)

    return vertices, np.asarray(filtered_faces, dtype=np.int64)


def edge_grids_to_points_and_segments(
    edge_points_world: np.ndarray,
    weld_tolerance: Optional[float] = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert per-edge point sequences into a global polyline graph.

    Returns:
        points: (N, 3) float32
        segments: (M, 2) int64, zero-based indices into points
    """
    if edge_points_world.ndim != 3 or edge_points_world.shape[-1] != 3:
        raise ValueError("edge_points_world must have shape (num_edges, num_points, 3)")

    indexer = _PointIndexer(weld_tolerance)
    segments: List[Tuple[int, int]] = []

    for polyline in edge_points_world:
        if polyline.shape[0] == 0:
            continue

        polyline_ids = [indexer.add(point) for point in polyline]
        for start, end in zip(polyline_ids[:-1], polyline_ids[1:]):
            if start != end:
                segments.append((int(start), int(end)))

    points = indexer.as_array()
    if not segments:
        return points, np.zeros((0, 2), dtype=np.int64)

    return points, np.asarray(segments, dtype=np.int64)


def extract_surface_and_edge_geometry(
    row: RowDict,
    weld_tolerance: Optional[float] = 1e-6,
) -> RowDict:
    """
    Build surface and edge geometry from one ABC row.

    The input can be either:
      - a raw parquet row (serialized byte fields), or
      - an already-deserialized row produced by deserialize_geometry_row().

    Returns a dictionary containing:
      - surface_vertices: (N, 3)
      - surface_faces: (M, 3)
      - edge_points: (K, 3)
      - edge_segments: (L, 2)
      - face_points_world: (num_faces, num_u, num_v, 3)
      - edge_points_world: (num_edges, num_points, 3)
      - face_edge_incidence: (num_faces, num_edges)
    """
    if not isinstance(row.get("face_points_normalized"), np.ndarray):
        row = deserialize_geometry_row(row)

    if not validate_geometry_row(row):
        raise ValueError("Row does not contain a valid ABC geometry sample")

    face_points_world = convert_normalized_face_points_to_world(
        row["face_bbox_world"],
        row["face_points_normalized"],
    )
    edge_points_world = convert_normalized_edge_points_to_world(
        row["edge_bbox_world"],
        row["edge_points_normalized"],
    )

    surface_vertices, surface_faces = surface_grids_to_vertices_and_faces(
        face_points_world,
        weld_tolerance=weld_tolerance,
    )
    edge_points, edge_segments = edge_grids_to_points_and_segments(
        edge_points_world,
        weld_tolerance=weld_tolerance,
    )

    return {
        "surface_vertices": surface_vertices,
        "surface_faces": surface_faces,
        "edge_points": edge_points,
        "edge_segments": edge_segments,
        "face_points_world": face_points_world.astype(np.float32),
        "edge_points_world": edge_points_world.astype(np.float32),
        "face_edge_incidence": row["face_edge_incidence"].astype(bool),
    }


def _write_vertices(handle, vertices: np.ndarray) -> None:
    for x, y, z in vertices:
        handle.write(f"v {x:.10g} {y:.10g} {z:.10g}\n")


def write_surface_obj(
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
    obj_path: PathLike,
) -> Path:
    """Write a triangle mesh to OBJ using `v` and `f` records."""
    obj_path = Path(obj_path)
    obj_path.parent.mkdir(parents=True, exist_ok=True)

    with obj_path.open("w", encoding="utf-8") as handle:
        handle.write("# AutoBrep surface mesh\n")
        _write_vertices(handle, surface_vertices)
        for a, b, c in np.asarray(surface_faces, dtype=np.int64):
            handle.write(f"f {a + 1} {b + 1} {c + 1}\n")

    return obj_path


def write_edge_obj(
    edge_points: np.ndarray,
    edge_segments: np.ndarray,
    obj_path: PathLike,
) -> Path:
    """Write edge polylines to OBJ using `v` and `l` records."""
    obj_path = Path(obj_path)
    obj_path.parent.mkdir(parents=True, exist_ok=True)

    with obj_path.open("w", encoding="utf-8") as handle:
        handle.write("# AutoBrep edge polylines\n")
        _write_vertices(handle, edge_points)
        for start, end in np.asarray(edge_segments, dtype=np.int64):
            handle.write(f"l {start + 1} {end + 1}\n")

    return obj_path


def write_surface_and_edge_objs(
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
    edge_points: np.ndarray,
    edge_segments: np.ndarray,
    surface_obj_path: PathLike,
    edge_obj_path: PathLike,
) -> Tuple[Path, Path]:
    """
    Write the surface mesh to one OBJ and the edge polyline graph to another OBJ.
    """
    surface_path = write_surface_obj(surface_vertices, surface_faces, surface_obj_path)
    edge_path = write_edge_obj(edge_points, edge_segments, edge_obj_path)
    return surface_path, edge_path


def export_row_to_obj_files(
    row: RowDict,
    surface_obj_path: PathLike,
    edge_obj_path: PathLike,
    weld_tolerance: Optional[float] = 1e-6,
) -> RowDict:
    """
    Convenience wrapper: row -> geometry arrays -> OBJ files.

    Returns the same geometry dictionary produced by
    extract_surface_and_edge_geometry().
    """
    geometry = extract_surface_and_edge_geometry(row, weld_tolerance=weld_tolerance)
    write_surface_and_edge_objs(
        geometry["surface_vertices"],
        geometry["surface_faces"],
        geometry["edge_points"],
        geometry["edge_segments"],
        surface_obj_path,
        edge_obj_path,
    )
    return geometry


class ABCGeometryDataModule(LightningDataModule):
    """
    Lightning-style data module that streams ABC parquet splits and returns
    geometry dictionaries instead of tokenized CAD sequences.

    Each yielded sample is the output of extract_surface_and_edge_geometry().
    """

    def __init__(
        self,
        data_root: PathLike,
        batch_size: int = 1,
        buffer_size: int = 0,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        prefetch_batches: int = 1,
        rows_per_arrow_batch: int = 4096,
        min_face: Optional[int] = None,
        max_face: Optional[int] = None,
        scaled_unique: Optional[bool] = None,
        limit_train: Optional[int] = None,
        limit_val: Optional[int] = None,
        limit_test: Optional[int] = None,
        weld_tolerance: Optional[float] = 1e-6,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_batches = prefetch_batches
        self.rows_per_arrow_batch = rows_per_arrow_batch
        self.min_face = min_face
        self.max_face = max_face
        self.scaled_unique = scaled_unique
        self.limit_train = limit_train
        self.limit_val = limit_val
        self.limit_test = limit_test
        self.weld_tolerance = weld_tolerance

        self._train_ds: Optional[ParquetRowIterable] = None
        self._val_ds: Optional[ParquetRowIterable] = None
        self._test_ds: Optional[ParquetRowIterable] = None

    @property
    def columns(self) -> List[str]:
        return self._dataset_columns()

    def _filter_expr(self) -> Optional[pc.Expression]:
        expr: Optional[pc.Expression] = None

        if self.min_face is not None:
            current = pc.field("num_faces_after_splitting") >= self.min_face
            expr = current if expr is None else expr & current
        if self.max_face is not None:
            current = pc.field("num_faces_after_splitting") <= self.max_face
            expr = current if expr is None else expr & current
        if self.scaled_unique is not None:
            current = pc.field("scaled_unique") == self.scaled_unique
            expr = current if expr is None else expr & current

        return expr

    def _dataset_columns(self) -> List[str]:
        columns = list(GEOMETRY_COLUMNS)
        if self.min_face is not None or self.max_face is not None:
            columns.append("num_faces_after_splitting")
        if self.scaled_unique is not None:
            columns.append("scaled_unique")
        return list(dict.fromkeys(columns))

    def _make_dataset(
        self,
        split: str,
        limit: Optional[int],
        shuffle_buffer_size: Optional[int],
    ) -> ParquetRowIterable:
        return ParquetRowIterable(
            paths=self.data_root / split,
            columns=self._dataset_columns(),
            filter_expr=self._filter_expr(),
            pre_filter=lambda row: True,
            deserialize_fn=deserialize_geometry_row,
            post_filter=validate_geometry_row,
            map_func=lambda row: extract_surface_and_edge_geometry(
                row,
                weld_tolerance=self.weld_tolerance,
            ),
            limit=limit,
            shuffle_buffer_size=shuffle_buffer_size,
            shuffle_seed=9876,
            batch_rows_read=self.rows_per_arrow_batch,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self._train_ds = self._make_dataset(
                split="train",
                limit=self.limit_train,
                shuffle_buffer_size=self.buffer_size,
            )
            self._val_ds = self._make_dataset(
                split="val",
                limit=self.limit_val,
                shuffle_buffer_size=None,
            )

        if stage in (None, "test"):
            self._test_ds = self._make_dataset(
                split="test",
                limit=self.limit_test,
                shuffle_buffer_size=None,
            )

    @staticmethod
    def collate_fn(batch: List[RowDict]) -> List[RowDict]:
        """
        Keep variable-sized geometry samples as a list instead of attempting to
        stack them into a dense tensor batch.
        """
        return batch

    def _dataloader_kwargs(self) -> Dict[str, Any]:
        prefetch_factor = None
        if self.num_workers > 0:
            prefetch_factor = max(1, int(self.prefetch_batches))

        return {
            "batch_size": self.batch_size,
            "drop_last": False,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": bool(self.persistent_workers)
            and self.num_workers > 0,
            "prefetch_factor": prefetch_factor,
            "collate_fn": self.collate_fn,
        }

    def train_dataloader(self) -> DataLoader:
        if self._train_ds is None:
            self.setup("fit")
        return DataLoader(self._train_ds, **self._dataloader_kwargs())

    def val_dataloader(self) -> DataLoader:
        if self._val_ds is None:
            self.setup("fit")
        return DataLoader(self._val_ds, **self._dataloader_kwargs())

    def test_dataloader(self) -> DataLoader:
        if self._test_ds is None:
            self.setup("test")
        return DataLoader(self._test_ds, **self._dataloader_kwargs())


__all__ = [
    "ABCGeometryDataModule",
    "GEOMETRY_COLUMNS",
    "ParquetRowIterable",
    "convert_normalized_edge_points_to_world",
    "convert_normalized_face_points_to_world",
    "deserialize_geometry_row",
    "edge_grids_to_points_and_segments",
    "export_row_to_obj_files",
    "extract_surface_and_edge_geometry",
    "surface_grids_to_vertices_and_faces",
    "validate_geometry_row",
    "write_edge_obj",
    "write_surface_and_edge_objs",
    "write_surface_obj",
]


if __name__ == "__main__":

    from pathlib import Path

    parquet_dir = "/home/vthamizharas/Documents/TRELLIS.2/datasets/AutoBrep_Dataset/"
    out_dir = Path("/home/vthamizharas/Documents/TRELLIS.2/brep_parquet_outputs/")
    out_dir.mkdir(exist_ok=True)

    dataset = ParquetRowIterable(
        paths=parquet_dir,
        columns=GEOMETRY_COLUMNS,
        deserialize_fn=deserialize_geometry_row,
        post_filter=validate_geometry_row,
        map_func=lambda row: extract_surface_and_edge_geometry(
            row, weld_tolerance=1e-6
        ),
        limit=10,
    )

    for i, geom in enumerate(dataset):
        surface_obj = out_dir / f"sample_{i:06d}_surface.obj"
        edge_obj = out_dir / f"sample_{i:06d}_edges.obj"

        write_surface_and_edge_objs(
            geom["surface_vertices"],
            geom["surface_faces"],
            geom["edge_points"],
            geom["edge_segments"],
            surface_obj,
            edge_obj,
        )

        print(f"wrote {surface_obj} and {edge_obj}")
        print("surface vertices:", geom["surface_vertices"].shape)
        print("surface faces   :", geom["surface_faces"].shape)
        print("edge points     :", geom["edge_points"].shape)
        print("edge segments   :", geom["edge_segments"].shape)
