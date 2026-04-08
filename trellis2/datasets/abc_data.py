from __future__ import annotations

"""
Utilities for streaming ABC-1M parquet rows and converting each row into:

- a triangulated surface mesh (vertices + triangle faces)
- a polyline edge graph (points + line segments)

This is a simplified replacement for the original tokenization-heavy abc_data.py.
It intentionally keeps the parquet streaming/deserialization path and removes the
CAD tokenization / augmentation logic that is not needed for geometry export.

Trim handling
-------------
The surface triangulation uses the incident face edges as trim wires in the face's
sampled UV domain. Each face edge polyline is projected to UV, ordered into loops,
split against the UV grid lines, and then each clipped UV cell piece is triangulated.
This avoids the keep/drop ambiguity that created holes or protruding triangles near
intersection boundaries.

If trim-loop reconstruction fails for a face, the mesher falls back to ``face_mask``
(if present), and finally to meshing the full rectangular UV patch.
"""

from pathlib import Path
import io
import math
import random
import warnings
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds
import torch
from torch.utils.data import DataLoader, IterableDataset
from shapely.prepared import prep

try:
    from pytorch_lightning import LightningDataModule
except ImportError:  # pragma: no cover - fallback for non-Lightning environments

    class LightningDataModule:  # type: ignore[override]
        """Lightweight fallback so this module can be used without Lightning."""

        pass


try:  # pragma: no cover - import guarded for environments without shapely
    import shapely
    from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon, box

    SHAPELY_AVAILABLE = True
except Exception:  # pragma: no cover - fallback path
    shapely = None  # type: ignore[assignment]
    GeometryCollection = Any  # type: ignore[misc,assignment]
    MultiPolygon = Any  # type: ignore[misc,assignment]
    Point = Any  # type: ignore[misc,assignment]
    Polygon = Any  # type: ignore[misc,assignment]
    box = None  # type: ignore[assignment]
    SHAPELY_AVAILABLE = False


PathLike = Union[str, Path]
RowDict = Dict[str, Any]

GEOMETRY_COLUMNS = [
    "face_points_normalized",
    "edge_points_normalized",
    "face_bbox_world",
    "edge_bbox_world",
    "face_edge_incidence",
    "face_mask",
]

import time
from functools import wraps


def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            print(f"{func.__name__} took {elapsed:.6f} seconds")

    return wrapper


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
        available_columns = set(dataset.schema.names)
        scanner_columns = [
            column for column in self.columns if column in available_columns
        ]
        scanner = dataset.scanner(
            columns=scanner_columns,
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
    data = {
        "face_points_normalized": deserialize_array(row["face_points_normalized"]),
        "edge_points_normalized": deserialize_array(row["edge_points_normalized"]),
        "face_bbox_world": deserialize_array(row["face_bbox_world"]),
        "edge_bbox_world": deserialize_array(row["edge_bbox_world"]),
        "face_edge_incidence": deserialize_array(row["face_edge_incidence"]).astype(
            bool
        ),
    }
    if "face_mask" in row and row["face_mask"] is not None:
        data["face_mask"] = deserialize_array(row["face_mask"])
    return data


def validate_geometry_row(row: RowDict) -> bool:
    """Basic shape checks after deserialization."""
    required = {
        "face_points_normalized",
        "edge_points_normalized",
        "face_bbox_world",
        "edge_bbox_world",
        "face_edge_incidence",
    }
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

    num_faces, _, _, _ = face_points.shape
    num_edges, _, _ = edge_points.shape

    if face_bbox.shape[0] != num_faces:
        return False
    if edge_bbox.shape[0] != num_edges:
        return False
    if face_edge_incidence.shape != (num_faces, num_edges):
        return False

    face_mask = row.get("face_mask")
    if face_mask is not None and not isinstance(face_mask, np.ndarray):
        return False

    return True


@measure_time
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


@measure_time
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
    if not np.all(np.isfinite(tri)):
        return False
    area_twice = np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
    return bool(area_twice > area_eps)


def _as_boolean_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if np.issubdtype(arr.dtype, np.floating):
        return arr > 0.5
    return arr.astype(bool)


def _coerce_face_mask(
    face_mask: Any,
    num_faces: int,
    num_u: int,
    num_v: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Normalize a stored face mask to either:
      - vertex_mask: (num_faces, num_u, num_v)
      - cell_mask:   (num_faces, num_u-1, num_v-1)
    """
    if face_mask is None:
        return None, None

    mask = np.asarray(face_mask)

    changed = True
    while changed:
        changed = False
        if mask.ndim >= 4 and mask.shape[-1] == 1:
            mask = mask[..., 0]
            changed = True
        if mask.ndim >= 4 and mask.shape[1] == 1:
            mask = mask[:, 0]
            changed = True

    mask = _as_boolean_mask(mask)

    if mask.shape == (num_faces, num_u, num_v):
        return mask, None
    if mask.shape == (num_faces, num_v, num_u):
        return np.transpose(mask, (0, 2, 1)), None

    if num_u > 1 and num_v > 1 and mask.shape == (num_faces, num_u - 1, num_v - 1):
        return None, mask
    if num_u > 1 and num_v > 1 and mask.shape == (num_faces, num_v - 1, num_u - 1):
        return None, np.transpose(mask, (0, 2, 1))

    if mask.ndim == 2 and mask.shape == (num_faces, num_u * num_v):
        return mask.reshape(num_faces, num_u, num_v), None
    if (
        num_u > 1
        and num_v > 1
        and mask.ndim == 2
        and mask.shape == (num_faces, (num_u - 1) * (num_v - 1))
    ):
        return None, mask.reshape(num_faces, num_u - 1, num_v - 1)

    if num_faces == 1 and mask.shape == (num_u, num_v):
        return mask.reshape(1, num_u, num_v), None
    if (
        num_faces == 1
        and num_u > 1
        and num_v > 1
        and mask.shape == (num_u - 1, num_v - 1)
    ):
        return None, mask.reshape(1, num_u - 1, num_v - 1)

    raise ValueError(
        f"Unsupported face_mask shape {mask.shape}; expected one of "
        f"({num_faces}, {num_u}, {num_v}) or ({num_faces}, {num_u - 1}, {num_v - 1})"
    )


def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (
        (np.asarray(a, dtype=np.float32) + np.asarray(b, dtype=np.float32)) * 0.5
    ).astype(np.float32)


def _append_polygon_as_fan(
    polygon_points: Sequence[np.ndarray],
    indexer: _PointIndexer,
    faces: List[Tuple[int, int, int]],
) -> None:
    if len(polygon_points) < 3:
        return

    ids = [indexer.add(np.asarray(point, dtype=np.float32)) for point in polygon_points]
    for idx in range(1, len(ids) - 1):
        faces.append((int(ids[0]), int(ids[idx]), int(ids[idx + 1])))


def _append_masked_quad(
    quad_points: Sequence[np.ndarray],
    inside: Sequence[bool],
    indexer: _PointIndexer,
    faces: List[Tuple[int, int, int]],
) -> None:
    """
    Triangulate one quad clipped by a binary vertex mask.
    """
    inside = [bool(flag) for flag in inside]
    code = (
        (1 if inside[0] else 0)
        | ((1 if inside[1] else 0) << 1)
        | ((1 if inside[2] else 0) << 2)
        | ((1 if inside[3] else 0) << 3)
    )

    if code == 0:
        return
    if code == 15:
        _append_polygon_as_fan(quad_points, indexer, faces)
        return

    if code in (5, 10):
        if code == 5:
            polys = [
                [
                    quad_points[0],
                    _midpoint(quad_points[0], quad_points[1]),
                    _midpoint(quad_points[3], quad_points[0]),
                ],
                [
                    quad_points[2],
                    _midpoint(quad_points[1], quad_points[2]),
                    _midpoint(quad_points[2], quad_points[3]),
                ],
            ]
        else:
            polys = [
                [
                    quad_points[1],
                    _midpoint(quad_points[0], quad_points[1]),
                    _midpoint(quad_points[1], quad_points[2]),
                ],
                [
                    quad_points[3],
                    _midpoint(quad_points[2], quad_points[3]),
                    _midpoint(quad_points[3], quad_points[0]),
                ],
            ]
        for polygon in polys:
            _append_polygon_as_fan(polygon, indexer, faces)
        return

    polygon: List[np.ndarray] = []
    for idx in range(4):
        next_idx = (idx + 1) % 4
        if inside[idx]:
            polygon.append(np.asarray(quad_points[idx], dtype=np.float32))
        if inside[idx] != inside[next_idx]:
            polygon.append(_midpoint(quad_points[idx], quad_points[next_idx]))

    _append_polygon_as_fan(polygon, indexer, faces)


def _closest_point_on_triangle(
    point: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Closest point on triangle and barycentric coordinates.

    Based on the region tests in *Real-Time Collision Detection*.
    """
    ab = b - a
    ac = c - a
    ap = point - a

    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a, np.array([1.0, 0.0, 0.0], dtype=np.float64)

    bp = point - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b, np.array([0.0, 1.0, 0.0], dtype=np.float64)

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        denom = d1 - d3
        v = d1 / denom if abs(denom) > 1e-12 else 0.0
        closest = a + v * ab
        return closest, np.array([1.0 - v, v, 0.0], dtype=np.float64)

    cp = point - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c, np.array([0.0, 0.0, 1.0], dtype=np.float64)

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        denom = d2 - d6
        w = d2 / denom if abs(denom) > 1e-12 else 0.0
        closest = a + w * ac
        return closest, np.array([1.0 - w, 0.0, w], dtype=np.float64)

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        denom = (d4 - d3) + (d5 - d6)
        w = (d4 - d3) / denom if abs(denom) > 1e-12 else 0.0
        closest = b + w * (c - b)
        return closest, np.array([0.0, 1.0 - w, w], dtype=np.float64)

    denom = va + vb + vc
    if abs(denom) <= 1e-12:
        return a, np.array([1.0, 0.0, 0.0], dtype=np.float64)

    inv_denom = 1.0 / denom
    v = vb * inv_denom
    w = vc * inv_denom
    u = 1.0 - v - w
    closest = a + v * ab + w * ac
    return closest, np.array([u, v, w], dtype=np.float64)


def _project_point_to_face_uv(
    point_world: np.ndarray,
    face_grid_world: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Project a world-space point onto the sampled face grid and return UV index-space
    coordinates in the range ``[0, num_u - 1] x [0, num_v - 1]``.
    """
    point_world = np.asarray(point_world, dtype=np.float64)
    num_u, num_v, _ = face_grid_world.shape

    best_uv = np.zeros(2, dtype=np.float64)
    best_dist_sq = float("inf")

    for u in range(num_u - 1):
        for v in range(num_v - 1):
            p00 = np.asarray(face_grid_world[u, v], dtype=np.float64)
            p10 = np.asarray(face_grid_world[u + 1, v], dtype=np.float64)
            p11 = np.asarray(face_grid_world[u + 1, v + 1], dtype=np.float64)
            p01 = np.asarray(face_grid_world[u, v + 1], dtype=np.float64)

            triangle_data = (
                (
                    p00,
                    p10,
                    p11,
                    np.array([u, v], dtype=np.float64),
                    np.array([u + 1, v], dtype=np.float64),
                    np.array([u + 1, v + 1], dtype=np.float64),
                ),
                (
                    p00,
                    p11,
                    p01,
                    np.array([u, v], dtype=np.float64),
                    np.array([u + 1, v + 1], dtype=np.float64),
                    np.array([u, v + 1], dtype=np.float64),
                ),
            )

            for a, b, c, uv_a, uv_b, uv_c in triangle_data:
                closest, bary = _closest_point_on_triangle(point_world, a, b, c)
                dist_sq = float(np.sum((closest - point_world) ** 2))
                if dist_sq < best_dist_sq:
                    best_dist_sq = dist_sq
                    best_uv = bary[0] * uv_a + bary[1] * uv_b + bary[2] * uv_c

    return best_uv, best_dist_sq


def _project_polyline_to_face_uv(
    edge_world_points: np.ndarray,
    face_grid_world: np.ndarray,
) -> np.ndarray:
    projected = []
    for point in np.asarray(edge_world_points, dtype=np.float64):
        if not np.all(np.isfinite(point)):
            continue
        uv, _ = _project_point_to_face_uv(point, face_grid_world)
        projected.append(uv)

    if not projected:
        return np.zeros((0, 2), dtype=np.float64)
    return np.stack(projected, axis=0)


def _clean_polyline(
    uv_points: Sequence[np.ndarray],
    xyz_points: Sequence[np.ndarray],
    tol: float = 1e-9,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    clean_uv: List[np.ndarray] = []
    clean_xyz: List[np.ndarray] = []

    for uv, xyz in zip(uv_points, xyz_points):
        uv_arr = np.asarray(uv, dtype=np.float64)
        xyz_arr = np.asarray(xyz, dtype=np.float64)
        if not np.all(np.isfinite(uv_arr)) or not np.all(np.isfinite(xyz_arr)):
            continue
        if clean_uv and float(np.linalg.norm(uv_arr - clean_uv[-1])) <= tol:
            continue
        clean_uv.append(uv_arr)
        clean_xyz.append(xyz_arr)

    return clean_uv, clean_xyz


def _cluster_points(
    points: Sequence[np.ndarray], tol: float
) -> Tuple[List[int], List[np.ndarray]]:
    centers: List[np.ndarray] = []
    counts: List[int] = []
    assignment: List[int] = []

    for point in points:
        point = np.asarray(point, dtype=np.float64)
        cluster_id = None
        for idx, center in enumerate(centers):
            if float(np.linalg.norm(point - center)) <= tol:
                cluster_id = idx
                break

        if cluster_id is None:
            centers.append(point.copy())
            counts.append(1)
            assignment.append(len(centers) - 1)
        else:
            centers[cluster_id] = (
                centers[cluster_id] * counts[cluster_id] + point
            ) / float(counts[cluster_id] + 1)
            counts[cluster_id] += 1
            assignment.append(cluster_id)

    return assignment, centers


def _snap_polyline_endpoints(
    uv_polylines: Sequence[Sequence[np.ndarray]],
    xyz_polylines: Sequence[Sequence[np.ndarray]],
    endpoint_tolerance: float,
) -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]], List[Tuple[int, int]]]:
    endpoints: List[np.ndarray] = []
    endpoint_xyz: List[np.ndarray] = []
    for uv_polyline, xyz_polyline in zip(uv_polylines, xyz_polylines):
        endpoints.extend([uv_polyline[0], uv_polyline[-1]])
        endpoint_xyz.extend([xyz_polyline[0], xyz_polyline[-1]])

    assignment, uv_centers = _cluster_points(endpoints, endpoint_tolerance)
    xyz_sums = [np.zeros(3, dtype=np.float64) for _ in uv_centers]
    xyz_counts = [0 for _ in uv_centers]
    for cluster_id, xyz in zip(assignment, endpoint_xyz):
        xyz_sums[cluster_id] += np.asarray(xyz, dtype=np.float64)
        xyz_counts[cluster_id] += 1
    xyz_centers = [
        xyz_sums[idx] / float(max(xyz_counts[idx], 1)) for idx in range(len(uv_centers))
    ]

    snapped_uv: List[List[np.ndarray]] = []
    snapped_xyz: List[List[np.ndarray]] = []
    node_pairs: List[Tuple[int, int]] = []

    for poly_idx, (uv_polyline, xyz_polyline) in enumerate(
        zip(uv_polylines, xyz_polylines)
    ):
        start_id = assignment[2 * poly_idx]
        end_id = assignment[2 * poly_idx + 1]

        uv_copy = [np.asarray(point, dtype=np.float64).copy() for point in uv_polyline]
        xyz_copy = [
            np.asarray(point, dtype=np.float64).copy() for point in xyz_polyline
        ]
        uv_copy[0] = uv_centers[start_id].copy()
        uv_copy[-1] = uv_centers[end_id].copy()
        xyz_copy[0] = xyz_centers[start_id].copy()
        xyz_copy[-1] = xyz_centers[end_id].copy()

        snapped_uv.append(uv_copy)
        snapped_xyz.append(xyz_copy)
        node_pairs.append((start_id, end_id))

    return snapped_uv, snapped_xyz, node_pairs


def _build_uv_loops_from_edges(
    uv_polylines: Sequence[np.ndarray],
    xyz_polylines: Sequence[np.ndarray],
    endpoint_tolerance: float,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    cleaned_pairs = [
        _clean_polyline(uv_polyline, xyz_polyline)
        for uv_polyline, xyz_polyline in zip(uv_polylines, xyz_polylines)
    ]
    uv_clean = [pair[0] for pair in cleaned_pairs if len(pair[0]) >= 2]
    xyz_clean = [pair[1] for pair in cleaned_pairs if len(pair[1]) >= 2]

    if not uv_clean:
        return []

    uv_clean, xyz_clean, node_pairs = _snap_polyline_endpoints(
        uv_clean,
        xyz_clean,
        endpoint_tolerance=endpoint_tolerance,
    )

    adjacency: Dict[int, List[Tuple[int, bool]]] = {}
    for edge_idx, (start_id, end_id) in enumerate(node_pairs):
        adjacency.setdefault(start_id, []).append((edge_idx, True))
        adjacency.setdefault(end_id, []).append((edge_idx, False))

    unused_edges = set(range(len(uv_clean)))
    loops: List[Tuple[np.ndarray, np.ndarray]] = []

    while unused_edges:
        edge_idx = next(iter(unused_edges))
        unused_edges.remove(edge_idx)

        start_node, end_node = node_pairs[edge_idx]
        loop_uv = [point.copy() for point in uv_clean[edge_idx]]
        loop_xyz = [point.copy() for point in xyz_clean[edge_idx]]
        current_node = end_node
        start_anchor = start_node

        safety = 0
        max_steps = max(4 * len(uv_clean), 16)
        while current_node != start_anchor and safety < max_steps:
            safety += 1
            candidates = [
                candidate
                for candidate in adjacency.get(current_node, [])
                if candidate[0] in unused_edges
            ]
            if not candidates:
                break

            next_edge_idx, at_start = candidates[0]
            unused_edges.remove(next_edge_idx)
            next_start_node, next_end_node = node_pairs[next_edge_idx]

            if at_start:
                next_uv = [point.copy() for point in uv_clean[next_edge_idx]]
                next_xyz = [point.copy() for point in xyz_clean[next_edge_idx]]
                current_node = next_end_node
            else:
                next_uv = [point.copy() for point in reversed(uv_clean[next_edge_idx])]
                next_xyz = [
                    point.copy() for point in reversed(xyz_clean[next_edge_idx])
                ]
                current_node = next_start_node

            loop_uv.extend(next_uv[1:])
            loop_xyz.extend(next_xyz[1:])

        if len(loop_uv) < 3:
            continue

        if float(np.linalg.norm(loop_uv[0] - loop_uv[-1])) > endpoint_tolerance:
            loop_uv.append(loop_uv[0].copy())
            loop_xyz.append(loop_xyz[0].copy())

        loops.append(
            (
                np.asarray(loop_uv, dtype=np.float64),
                np.asarray(loop_xyz, dtype=np.float64),
            )
        )

    return loops


def _split_polyline_at_grid_lines(
    uv_points: np.ndarray,
    xyz_points: np.ndarray,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Insert vertices where a UV polyline crosses integer grid lines, so later cell
    clipping does not have to invent new trim-boundary vertices.
    """
    uv_points = np.asarray(uv_points, dtype=np.float64)
    xyz_points = np.asarray(xyz_points, dtype=np.float64)

    if uv_points.shape[0] != xyz_points.shape[0]:
        raise ValueError("UV and XYZ polylines must have the same number of points")

    out_uv: List[np.ndarray] = []
    out_xyz: List[np.ndarray] = []

    for idx in range(len(uv_points) - 1):
        uv0 = uv_points[idx]
        uv1 = uv_points[idx + 1]
        xyz0 = xyz_points[idx]
        xyz1 = xyz_points[idx + 1]

        params = [0.0, 1.0]
        du = float(uv1[0] - uv0[0])
        dv = float(uv1[1] - uv0[1])

        if abs(du) > eps:
            lo = float(min(uv0[0], uv1[0]))
            hi = float(max(uv0[0], uv1[0]))
            for grid_u in range(math.floor(lo + eps) + 1, math.ceil(hi - eps)):
                t = (grid_u - uv0[0]) / du
                if eps < t < 1.0 - eps:
                    params.append(float(t))

        if abs(dv) > eps:
            lo = float(min(uv0[1], uv1[1]))
            hi = float(max(uv0[1], uv1[1]))
            for grid_v in range(math.floor(lo + eps) + 1, math.ceil(hi - eps)):
                t = (grid_v - uv0[1]) / dv
                if eps < t < 1.0 - eps:
                    params.append(float(t))

        params = sorted(set(round(float(param), 12) for param in params))

        for param in params[:-1]:
            uv = (1.0 - param) * uv0 + param * uv1
            xyz = (1.0 - param) * xyz0 + param * xyz1
            if out_uv and float(np.linalg.norm(uv - out_uv[-1])) <= eps:
                continue
            out_uv.append(uv)
            out_xyz.append(xyz)

    if not out_uv or float(np.linalg.norm(uv_points[-1] - out_uv[-1])) > eps:
        out_uv.append(uv_points[-1].copy())
        out_xyz.append(xyz_points[-1].copy())

    return np.asarray(out_uv, dtype=np.float64), np.asarray(out_xyz, dtype=np.float64)


def _signed_area_2d(loop_uv: np.ndarray) -> float:
    loop_uv = np.asarray(loop_uv, dtype=np.float64)
    if loop_uv.shape[0] < 3:
        return 0.0
    x = loop_uv[:, 0]
    y = loop_uv[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def _build_trim_region_from_loops(
    uv_loops: Sequence[np.ndarray],
) -> Optional[Any]:
    if not SHAPELY_AVAILABLE:
        return None

    rings: List[Dict[str, Any]] = []
    for loop_uv in uv_loops:
        loop_uv = np.asarray(loop_uv, dtype=np.float64)
        if loop_uv.shape[0] < 4:
            continue
        if float(np.linalg.norm(loop_uv[0] - loop_uv[-1])) > 1e-9:
            loop_uv = np.concatenate([loop_uv, loop_uv[:1]], axis=0)

        polygon = Polygon(loop_uv)
        if polygon.is_empty or float(polygon.area) <= 1e-12:
            continue
        polygon = polygon.buffer(0)
        if polygon.is_empty or float(polygon.area) <= 1e-12:
            continue

        rings.append(
            {
                "coords": loop_uv,
                "poly": polygon,
                "area": abs(_signed_area_2d(loop_uv)),
            }
        )

    if not rings:
        return None

    rings.sort(key=lambda ring: ring["area"], reverse=True)

    for ring in rings:
        representative_point = ring["poly"].representative_point()
        depth = 0
        for container in rings:
            if container["area"] <= ring["area"] + 1e-12:
                continue
            if container["poly"].contains(representative_point):
                depth += 1
        ring["depth"] = depth

    outer_indices = [idx for idx, ring in enumerate(rings) if ring["depth"] % 2 == 0]
    if not outer_indices:
        return None

    holes_for_outer: Dict[int, List[np.ndarray]] = {idx: [] for idx in outer_indices}
    for ring_idx, ring in enumerate(rings):
        if ring["depth"] % 2 == 0:
            continue

        representative_point = ring["poly"].representative_point()
        candidate_outers = [
            outer_idx
            for outer_idx in outer_indices
            if rings[outer_idx]["poly"].contains(representative_point)
        ]
        if not candidate_outers:
            continue

        owner_idx = min(candidate_outers, key=lambda idx: rings[idx]["area"])
        holes_for_outer[owner_idx].append(ring["coords"])

    polygons = []
    for outer_idx in outer_indices:
        outer_ring = rings[outer_idx]["coords"]
        polygon = Polygon(outer_ring, holes_for_outer[outer_idx])
        polygon = polygon.buffer(0)
        if not polygon.is_empty and float(polygon.area) > 1e-12:
            polygons.append(polygon)

    if not polygons:
        return None

    if len(polygons) == 1:
        return polygons[0]
    return shapely.union_all(polygons)


def _round_uv_key(uv: np.ndarray, tolerance: float) -> Tuple[int, int]:
    scale = 1.0 / float(max(tolerance, 1e-12))
    uv = np.asarray(uv, dtype=np.float64)
    rounded = np.rint(uv * scale).astype(np.int64)
    return int(rounded[0]), int(rounded[1])


def _build_boundary_lookup(
    uv_loops: Sequence[np.ndarray],
    xyz_loops: Sequence[np.ndarray],
    tolerance: float,
) -> Dict[Tuple[int, int], np.ndarray]:
    xyz_sum: Dict[Tuple[int, int], np.ndarray] = {}
    xyz_count: Dict[Tuple[int, int], int] = {}

    for uv_loop, xyz_loop in zip(uv_loops, xyz_loops):
        for uv, xyz in zip(uv_loop, xyz_loop):
            key = _round_uv_key(uv, tolerance)
            xyz_sum[key] = xyz_sum.get(key, np.zeros(3, dtype=np.float64)) + np.asarray(
                xyz, dtype=np.float64
            )
            xyz_count[key] = xyz_count.get(key, 0) + 1

    return {
        key: (xyz_sum[key] / float(max(xyz_count[key], 1))).astype(np.float32)
        for key in xyz_sum
    }


def _extract_polygon_geometries(geometry: Any) -> List[Any]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return [geom for geom in geometry.geoms if not geom.is_empty]
    if isinstance(geometry, GeometryCollection):
        polygons: List[Any] = []
        for geom in geometry.geoms:
            polygons.extend(_extract_polygon_geometries(geom))
        return polygons
    return []


def _triangle_polygons_from_piece(piece: Any) -> List[Any]:
    if not SHAPELY_AVAILABLE or piece.is_empty:
        return []

    piece = piece.buffer(0)
    if piece.is_empty or float(piece.area) <= 1e-12:
        return []

    if hasattr(shapely, "constrained_delaunay_triangles"):
        triangles = shapely.constrained_delaunay_triangles(piece)
        return [geom for geom in triangles.geoms if not geom.is_empty]

    return [tri for tri in shapely.ops.triangulate(piece) if piece.covers(tri)]


def _bilinear_point_in_cell(
    face_grid_world: np.ndarray,
    cell_u: int,
    cell_v: int,
    uv: np.ndarray,
) -> np.ndarray:
    p00 = np.asarray(face_grid_world[cell_u, cell_v], dtype=np.float64)
    p10 = np.asarray(face_grid_world[cell_u + 1, cell_v], dtype=np.float64)
    p11 = np.asarray(face_grid_world[cell_u + 1, cell_v + 1], dtype=np.float64)
    p01 = np.asarray(face_grid_world[cell_u, cell_v + 1], dtype=np.float64)

    s = float(np.clip(uv[0] - cell_u, 0.0, 1.0))
    t = float(np.clip(uv[1] - cell_v, 0.0, 1.0))

    point = (
        (1.0 - s) * (1.0 - t) * p00
        + s * (1.0 - t) * p10
        + s * t * p11
        + (1.0 - s) * t * p01
    )
    return point.astype(np.float32)


def _uv_to_world_for_cell_vertex(
    face_grid_world: np.ndarray,
    cell_u: int,
    cell_v: int,
    uv: np.ndarray,
    boundary_lookup: Dict[Tuple[int, int], np.ndarray],
    uv_lookup_tolerance: float,
) -> np.ndarray:
    key = _round_uv_key(uv, uv_lookup_tolerance)
    if key in boundary_lookup:
        return np.asarray(boundary_lookup[key], dtype=np.float32)
    return _bilinear_point_in_cell(face_grid_world, cell_u, cell_v, uv)


def _triangulate_full_face_grid_vectorized(
    face_grid_world: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    num_u, num_v, _ = face_grid_world.shape

    vertices = face_grid_world.reshape(-1, 3).astype(np.float32, copy=False)

    ids = np.arange(num_u * num_v, dtype=np.int64).reshape(num_u, num_v)

    tri1 = np.stack(
        [ids[:-1, :-1], ids[1:, :-1], ids[1:, 1:]],
        axis=-1,
    ).reshape(-1, 3)

    tri2 = np.stack(
        [ids[:-1, :-1], ids[1:, 1:], ids[:-1, 1:]],
        axis=-1,
    ).reshape(-1, 3)

    faces = np.concatenate([tri1, tri2], axis=0)

    tri_pts = vertices[faces]
    valid = np.isfinite(tri_pts).all(axis=(1, 2))
    faces = faces[valid]

    return vertices, faces


def _append_full_face_grid(
    face_grid_world: np.ndarray,
    indexer: _PointIndexer,
    faces: List[Tuple[int, int, int]],
) -> None:
    num_u, num_v, _ = face_grid_world.shape
    face_vertex_ids = np.empty((num_u, num_v), dtype=np.int64)

    for u in range(num_u):
        for v in range(num_v):
            face_vertex_ids[u, v] = indexer.add(face_grid_world[u, v])

    for u in range(num_u - 1):
        for v in range(num_v - 1):
            a = int(face_vertex_ids[u, v])
            b = int(face_vertex_ids[u + 1, v])
            c = int(face_vertex_ids[u + 1, v + 1])
            d = int(face_vertex_ids[u, v + 1])

            faces.append((a, b, c))
            faces.append((a, c, d))


def _append_face_from_mask(
    face_grid_world: np.ndarray,
    face_vertex_mask: Optional[np.ndarray],
    face_cell_mask: Optional[np.ndarray],
    indexer: _PointIndexer,
    faces: List[Tuple[int, int, int]],
) -> None:
    num_u, num_v, _ = face_grid_world.shape

    for u in range(num_u - 1):
        for v in range(num_v - 1):
            if face_cell_mask is not None and not bool(face_cell_mask[u, v]):
                continue

            quad = [
                face_grid_world[u, v],
                face_grid_world[u + 1, v],
                face_grid_world[u + 1, v + 1],
                face_grid_world[u, v + 1],
            ]

            if not np.all(np.isfinite(np.asarray(quad))):
                continue

            if face_vertex_mask is None:
                _append_polygon_as_fan(quad, indexer, faces)
                continue

            inside = [
                bool(face_vertex_mask[u, v]),
                bool(face_vertex_mask[u + 1, v]),
                bool(face_vertex_mask[u + 1, v + 1]),
                bool(face_vertex_mask[u, v + 1]),
            ]
            _append_masked_quad(quad, inside, indexer, faces)


def _build_face_trim_region(
    face_grid_world: np.ndarray,
    incident_edge_points_world: np.ndarray,
    uv_endpoint_tolerance: float,
    uv_lookup_tolerance: float,
) -> Tuple[Optional[Any], Dict[Tuple[int, int], np.ndarray]]:
    if not SHAPELY_AVAILABLE:
        return None, {}

    projected_uv_polylines: List[np.ndarray] = []
    world_polylines: List[np.ndarray] = []

    for edge_polyline_world in np.asarray(incident_edge_points_world, dtype=np.float64):
        edge_polyline_world = edge_polyline_world[
            np.all(np.isfinite(edge_polyline_world), axis=1)
        ]
        if edge_polyline_world.shape[0] < 2:
            continue

        projected_uv = _project_polyline_to_face_uv(
            edge_polyline_world, face_grid_world
        )
        if projected_uv.shape[0] < 2:
            continue

        projected_uv_polylines.append(projected_uv)
        world_polylines.append(edge_polyline_world)

    if not projected_uv_polylines:
        return None, {}

    uv_loops_xyz = _build_uv_loops_from_edges(
        projected_uv_polylines,
        world_polylines,
        endpoint_tolerance=uv_endpoint_tolerance,
    )
    if not uv_loops_xyz:
        return None, {}

    split_uv_loops: List[np.ndarray] = []
    split_xyz_loops: List[np.ndarray] = []

    for uv_loop, xyz_loop in uv_loops_xyz:
        split_uv, split_xyz = _split_polyline_at_grid_lines(uv_loop, xyz_loop)
        clean_uv, clean_xyz = _clean_polyline(split_uv, split_xyz)
        if len(clean_uv) < 4:
            continue

        split_uv_array = np.asarray(clean_uv, dtype=np.float64)
        split_xyz_array = np.asarray(clean_xyz, dtype=np.float64)

        if (
            float(np.linalg.norm(split_uv_array[0] - split_uv_array[-1]))
            > uv_endpoint_tolerance
        ):
            split_uv_array = np.concatenate(
                [split_uv_array, split_uv_array[:1]], axis=0
            )
            split_xyz_array = np.concatenate(
                [split_xyz_array, split_xyz_array[:1]], axis=0
            )

        split_uv_loops.append(split_uv_array)
        split_xyz_loops.append(split_xyz_array)

    if not split_uv_loops:
        return None, {}

    trim_region = _build_trim_region_from_loops(split_uv_loops)
    if trim_region is None or trim_region.is_empty or float(trim_region.area) <= 1e-12:
        return None, {}

    num_u, num_v, _ = face_grid_world.shape
    domain = box(0.0, 0.0, float(num_u - 1), float(num_v - 1))
    trim_region = trim_region.intersection(domain).buffer(0)
    if trim_region.is_empty or float(trim_region.area) <= 1e-12:
        return None, {}

    boundary_lookup = _build_boundary_lookup(
        split_uv_loops,
        split_xyz_loops,
        tolerance=uv_lookup_tolerance,
    )
    return trim_region, boundary_lookup


def _append_face_from_trim_region(
    face_grid_world: np.ndarray,
    trim_region_uv: Any,
    boundary_lookup: Dict[Tuple[int, int], np.ndarray],
    indexer: _PointIndexer,
    faces: List[Tuple[int, int, int]],
    uv_lookup_tolerance: float,
) -> None:
    num_u, num_v, _ = face_grid_world.shape

    for cell_u in range(num_u - 1):
        for cell_v in range(num_v - 1):
            cell_box = box(
                float(cell_u), float(cell_v), float(cell_u + 1), float(cell_v + 1)
            )
            clipped = trim_region_uv.intersection(cell_box)
            piece_polygons = _extract_polygon_geometries(clipped)

            if not piece_polygons:
                continue

            cell_ref_normal = np.cross(
                face_grid_world[cell_u + 1, cell_v] - face_grid_world[cell_u, cell_v],
                face_grid_world[cell_u, cell_v + 1] - face_grid_world[cell_u, cell_v],
            )

            for piece in piece_polygons:
                if piece.is_empty or float(piece.area) <= 1e-12:
                    continue

                for tri_poly in _triangle_polygons_from_piece(piece):
                    coords = np.asarray(tri_poly.exterior.coords[:-1], dtype=np.float64)
                    if coords.shape != (3, 2):
                        continue

                    tri_points = np.stack(
                        [
                            _uv_to_world_for_cell_vertex(
                                face_grid_world,
                                cell_u,
                                cell_v,
                                uv=coord,
                                boundary_lookup=boundary_lookup,
                                uv_lookup_tolerance=uv_lookup_tolerance,
                            )
                            for coord in coords
                        ],
                        axis=0,
                    ).astype(np.float32)

                    if not np.all(np.isfinite(tri_points)):
                        continue

                    tri_ids = [indexer.add(point) for point in tri_points]

                    tri_normal = np.cross(
                        tri_points[1] - tri_points[0],
                        tri_points[2] - tri_points[0],
                    )
                    if float(np.dot(tri_normal, cell_ref_normal)) < 0.0:
                        tri_ids[1], tri_ids[2] = tri_ids[2], tri_ids[1]

                    faces.append((int(tri_ids[0]), int(tri_ids[1]), int(tri_ids[2])))


def _append_face_from_trim_region_fast(
    face_grid_world: np.ndarray,
    trim_region_uv: Any,
    boundary_lookup: Dict[Tuple[int, int], np.ndarray],
    indexer: _PointIndexer,
    faces: List[Tuple[int, int, int]],
    uv_lookup_tolerance: float,
) -> None:
    num_u, num_v, _ = face_grid_world.shape

    minx, miny, maxx, maxy = trim_region_uv.bounds
    u0 = max(0, int(math.floor(minx)))
    v0 = max(0, int(math.floor(miny)))
    u1 = min(num_u - 1, int(math.ceil(maxx)))
    v1 = min(num_v - 1, int(math.ceil(maxy)))

    prepared = prep(trim_region_uv)

    for cell_u in range(u0, u1):
        for cell_v in range(v0, v1):
            cell_box = box(
                float(cell_u), float(cell_v), float(cell_u + 1), float(cell_v + 1)
            )

            if not prepared.intersects(cell_box):
                continue

            clipped = trim_region_uv.intersection(cell_box)
            piece_polygons = _extract_polygon_geometries(clipped)
            if not piece_polygons:
                continue

            cell_ref_normal = np.cross(
                face_grid_world[cell_u + 1, cell_v] - face_grid_world[cell_u, cell_v],
                face_grid_world[cell_u, cell_v + 1] - face_grid_world[cell_u, cell_v],
            )

            for piece in piece_polygons:
                if piece.is_empty or float(piece.area) <= 1e-12:
                    continue

                for tri_poly in _triangle_polygons_from_piece(piece):
                    coords = np.asarray(tri_poly.exterior.coords[:-1], dtype=np.float64)
                    if coords.shape != (3, 2):
                        continue

                    tri_points = np.stack(
                        [
                            _uv_to_world_for_cell_vertex(
                                face_grid_world,
                                cell_u,
                                cell_v,
                                uv=coord,
                                boundary_lookup=boundary_lookup,
                                uv_lookup_tolerance=uv_lookup_tolerance,
                            )
                            for coord in coords
                        ],
                        axis=0,
                    ).astype(np.float32)

                    if not np.all(np.isfinite(tri_points)):
                        continue

                    tri_ids = [indexer.add(point) for point in tri_points]

                    tri_normal = np.cross(
                        tri_points[1] - tri_points[0],
                        tri_points[2] - tri_points[0],
                    )
                    if float(np.dot(tri_normal, cell_ref_normal)) < 0.0:
                        tri_ids[1], tri_ids[2] = tri_ids[2], tri_ids[1]

                    faces.append((int(tri_ids[0]), int(tri_ids[1]), int(tri_ids[2])))


def _weld_vertices_vectorized(
    vertices: np.ndarray,
    faces: np.ndarray,
    weld_tolerance: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    if vertices.shape[0] == 0:
        return vertices.astype(np.float32), faces.astype(np.int64)

    if weld_tolerance is None:
        welded_vertices = vertices.astype(np.float32, copy=False)
        welded_faces = faces.astype(np.int64, copy=False)
    else:
        keys = np.round(vertices / weld_tolerance).astype(np.int64)
        _, unique_idx, inverse = np.unique(
            keys, axis=0, return_index=True, return_inverse=True
        )
        welded_vertices = vertices[unique_idx].astype(np.float32, copy=False)
        welded_faces = inverse[faces]

    tri = welded_vertices[welded_faces]
    area2 = np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
        axis=1,
    )

    keep = (
        (welded_faces[:, 0] != welded_faces[:, 1])
        & (welded_faces[:, 1] != welded_faces[:, 2])
        & (welded_faces[:, 0] != welded_faces[:, 2])
        & np.isfinite(area2)
        & (area2 > 1e-12)
    )

    return welded_vertices, welded_faces[keep].astype(np.int64)


def _mesh_one_face(
    face_grid_world: np.ndarray,
    face_vertex_mask: Optional[np.ndarray],
    face_cell_mask: Optional[np.ndarray],
    incident_edge_points_world: Optional[np.ndarray],
    weld_tolerance: Optional[float],
    uv_endpoint_tolerance: float,
    uv_lookup_tolerance: float,
    use_trim_wires: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    # Fast path: no trim, no mask
    if not use_trim_wires and face_vertex_mask is None and face_cell_mask is None:
        return _triangulate_full_face_grid_vectorized(face_grid_world)

    local_indexer = _PointIndexer(weld_tolerance)
    local_faces: List[Tuple[int, int, int]] = []

    used_trim_region = False
    if (
        use_trim_wires
        and incident_edge_points_world is not None
        and len(incident_edge_points_world) > 0
    ):
        trim_region, boundary_lookup = _build_face_trim_region(
            face_grid_world=face_grid_world,
            incident_edge_points_world=incident_edge_points_world,
            uv_endpoint_tolerance=uv_endpoint_tolerance,
            uv_lookup_tolerance=uv_lookup_tolerance,
        )
        if trim_region is not None:
            _append_face_from_trim_region_fast(
                face_grid_world=face_grid_world,
                trim_region_uv=trim_region,
                boundary_lookup=boundary_lookup,
                indexer=local_indexer,
                faces=local_faces,
                uv_lookup_tolerance=uv_lookup_tolerance,
            )
            used_trim_region = True

    if not used_trim_region:
        if face_vertex_mask is not None or face_cell_mask is not None:
            _append_face_from_mask(
                face_grid_world=face_grid_world,
                face_vertex_mask=face_vertex_mask,
                face_cell_mask=face_cell_mask,
                indexer=local_indexer,
                faces=local_faces,
            )
        else:
            return _triangulate_full_face_grid_vectorized(face_grid_world)

    local_vertices = local_indexer.as_array()
    if not local_faces:
        return local_vertices, np.zeros((0, 3), dtype=np.int64)

    return local_vertices, np.asarray(local_faces, dtype=np.int64)


@measure_time
def surface_grids_to_vertices_and_faces_fast(
    face_points_world: np.ndarray,
    edge_points_world: Optional[np.ndarray] = None,
    face_edge_incidence: Optional[np.ndarray] = None,
    face_mask: Optional[np.ndarray] = None,
    weld_tolerance: Optional[float] = 1e-6,
    uv_endpoint_tolerance: float = 0.25,
    uv_lookup_tolerance: float = 1e-6,
    use_trim_wires: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    if face_points_world.ndim != 4 or face_points_world.shape[-1] != 3:
        raise ValueError(
            "face_points_world must have shape (num_faces, num_u, num_v, 3)"
        )

    num_faces, num_u, num_v, _ = face_points_world.shape

    vertex_mask = None
    cell_mask = None
    if face_mask is not None:
        try:
            vertex_mask, cell_mask = _coerce_face_mask(
                face_mask, num_faces=num_faces, num_u=num_u, num_v=num_v
            )
        except ValueError:
            vertex_mask, cell_mask = None, None

    can_use_trim = (
        use_trim_wires
        and SHAPELY_AVAILABLE
        and edge_points_world is not None
        and face_edge_incidence is not None
        and isinstance(edge_points_world, np.ndarray)
        and isinstance(face_edge_incidence, np.ndarray)
        and edge_points_world.ndim == 3
        and face_edge_incidence.shape == (num_faces, edge_points_world.shape[0])
    )

    vertex_chunks = []
    face_chunks = []
    vertex_offset = 0

    for face_idx, face_grid_world in enumerate(face_points_world):
        face_vertex_mask = None if vertex_mask is None else vertex_mask[face_idx]
        face_cell_mask = None if cell_mask is None else cell_mask[face_idx]

        incident_edge_points = None
        if can_use_trim:
            incident_edge_indices = np.where(face_edge_incidence[face_idx])[0]
            if incident_edge_indices.size > 0:
                incident_edge_points = edge_points_world[incident_edge_indices]

        local_vertices, local_faces = _mesh_one_face(
            face_grid_world=face_grid_world,
            face_vertex_mask=face_vertex_mask,
            face_cell_mask=face_cell_mask,
            incident_edge_points_world=incident_edge_points,
            weld_tolerance=weld_tolerance,
            uv_endpoint_tolerance=uv_endpoint_tolerance,
            uv_lookup_tolerance=uv_lookup_tolerance,
            use_trim_wires=can_use_trim,
        )

        if local_vertices.shape[0] == 0:
            continue

        vertex_chunks.append(local_vertices.astype(np.float32, copy=False))
        if local_faces.shape[0] > 0:
            face_chunks.append(local_faces + vertex_offset)

        vertex_offset += local_vertices.shape[0]

    if not vertex_chunks:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64)

    vertices = np.concatenate(vertex_chunks, axis=0)
    faces = (
        np.concatenate(face_chunks, axis=0)
        if face_chunks
        else np.zeros((0, 3), dtype=np.int64)
    )

    return _weld_vertices_vectorized(vertices, faces, weld_tolerance)


@measure_time
def surface_grids_to_vertices_and_faces(
    face_points_world: np.ndarray,
    edge_points_world: Optional[np.ndarray] = None,
    face_edge_incidence: Optional[np.ndarray] = None,
    face_mask: Optional[np.ndarray] = None,
    weld_tolerance: Optional[float] = 1e-6,
    uv_endpoint_tolerance: float = 0.25,
    uv_lookup_tolerance: float = 1e-6,
    use_trim_wires: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Triangulate per-face point grids into a global triangle mesh.

    Priority order per face:
      1. trim-loop meshing from incident edges in UV
      2. face_mask-based meshing (if available)
      3. full rectangular grid triangulation

    Returns:
        vertices: (N, 3) float32
        faces: (M, 3) int64, zero-based indices into vertices
    """
    if face_points_world.ndim != 4 or face_points_world.shape[-1] != 3:
        raise ValueError(
            "face_points_world must have shape (num_faces, num_u, num_v, 3)"
        )

    num_faces, num_u, num_v, _ = face_points_world.shape
    vertex_mask = None
    cell_mask = None
    if face_mask is not None:
        try:
            vertex_mask, cell_mask = _coerce_face_mask(
                face_mask, num_faces=num_faces, num_u=num_u, num_v=num_v
            )
        except ValueError:
            vertex_mask, cell_mask = None, None

    use_trim_wires = (
        use_trim_wires
        and SHAPELY_AVAILABLE
        and edge_points_world is not None
        and face_edge_incidence is not None
        and isinstance(edge_points_world, np.ndarray)
        and isinstance(face_edge_incidence, np.ndarray)
        and edge_points_world.ndim == 3
        and face_edge_incidence.shape == (num_faces, edge_points_world.shape[0])
    )

    if (
        edge_points_world is not None or face_edge_incidence is not None
    ) and not SHAPELY_AVAILABLE:
        warnings.warn(
            "shapely is not available; falling back from trim-wire meshing to face_mask/full-grid meshing.",
            RuntimeWarning,
            stacklevel=2,
        )

    indexer = _PointIndexer(weld_tolerance)
    faces: List[Tuple[int, int, int]] = []

    for face_idx, face_grid_world in enumerate(face_points_world):
        used_trim_region = False

        if use_trim_wires:
            incident_edge_indices = np.where(face_edge_incidence[face_idx])[0]
            if incident_edge_indices.size > 0:
                trim_region, boundary_lookup = _build_face_trim_region(
                    face_grid_world=face_grid_world,
                    incident_edge_points_world=edge_points_world[incident_edge_indices],
                    uv_endpoint_tolerance=uv_endpoint_tolerance,
                    uv_lookup_tolerance=uv_lookup_tolerance,
                )
                if trim_region is not None:
                    _append_face_from_trim_region(
                        face_grid_world=face_grid_world,
                        trim_region_uv=trim_region,
                        boundary_lookup=boundary_lookup,
                        indexer=indexer,
                        faces=faces,
                        uv_lookup_tolerance=uv_lookup_tolerance,
                    )
                    used_trim_region = True

        if used_trim_region:
            continue

        face_vertex_mask = None if vertex_mask is None else vertex_mask[face_idx]
        face_cell_mask = None if cell_mask is None else cell_mask[face_idx]
        if face_vertex_mask is not None or face_cell_mask is not None:
            _append_face_from_mask(
                face_grid_world=face_grid_world,
                face_vertex_mask=face_vertex_mask,
                face_cell_mask=face_cell_mask,
                indexer=indexer,
                faces=faces,
            )
            continue

        _append_full_face_grid(
            face_grid_world=face_grid_world,
            indexer=indexer,
            faces=faces,
        )

    vertices = indexer.as_array()
    filtered_faces = [
        face for face in faces if _non_degenerate_triangle(*face, vertices)
    ]

    if not filtered_faces:
        return vertices, np.zeros((0, 3), dtype=np.int64)

    return vertices, np.asarray(filtered_faces, dtype=np.int64)


@measure_time
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

        polyline_ids = [
            indexer.add(point) for point in polyline if np.all(np.isfinite(point))
        ]
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
    uv_endpoint_tolerance: float = 0.25,
    uv_lookup_tolerance: float = 1e-6,
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
      - face_mask: optional raw face trim mask
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

    surface_vertices, surface_faces = surface_grids_to_vertices_and_faces_fast(
        face_points_world=face_points_world,
        edge_points_world=edge_points_world,
        face_edge_incidence=row["face_edge_incidence"].astype(bool),
        face_mask=row.get("face_mask"),
        weld_tolerance=weld_tolerance,
        uv_endpoint_tolerance=uv_endpoint_tolerance,
        uv_lookup_tolerance=uv_lookup_tolerance,
    )
    # surface_vertices, surface_faces = None, None

    # edge_points, edge_segments = edge_grids_to_points_and_segments(
    #     edge_points_world,
    #     weld_tolerance=weld_tolerance,
    # )
    edge_points, edge_segments = None, None

    output = {
        "surface_vertices": surface_vertices,
        "surface_faces": surface_faces,
        "edge_points": edge_points,
        "edge_segments": edge_segments,
        "face_points_world": face_points_world.astype(np.float32),
        "edge_points_world": edge_points_world.astype(np.float32),
        "face_edge_incidence": row["face_edge_incidence"].astype(bool),
    }
    if row.get("face_mask") is not None:
        output["face_mask"] = np.asarray(row["face_mask"])
    return output


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
    uv_endpoint_tolerance: float = 0.25,
    uv_lookup_tolerance: float = 1e-6,
) -> RowDict:
    """
    Convenience wrapper: row -> geometry arrays -> OBJ files.

    Returns the same geometry dictionary produced by
    extract_surface_and_edge_geometry().
    """
    geometry = extract_surface_and_edge_geometry(
        row,
        weld_tolerance=weld_tolerance,
        uv_endpoint_tolerance=uv_endpoint_tolerance,
        uv_lookup_tolerance=uv_lookup_tolerance,
    )
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
        uv_endpoint_tolerance: float = 0.25,
        uv_lookup_tolerance: float = 1e-6,
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
        self.uv_endpoint_tolerance = uv_endpoint_tolerance
        self.uv_lookup_tolerance = uv_lookup_tolerance

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
                uv_endpoint_tolerance=self.uv_endpoint_tolerance,
                uv_lookup_tolerance=self.uv_lookup_tolerance,
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
    "deserialize_array",
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
