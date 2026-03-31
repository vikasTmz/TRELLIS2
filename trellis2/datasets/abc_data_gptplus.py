from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
except ImportError:  # pragma: no cover - depends on local environment
    pc = None
    ds = None

try:
    from autobrep.data.serialize import deserialize_array
except ImportError:

    def deserialize_array(serialized: bytes) -> np.ndarray:
        """Fallback NumPy blob deserializer used by the ABC parquet files."""
        memfile = io.BytesIO()
        memfile.write(serialized)
        memfile.seek(0)
        return np.load(memfile, allow_pickle=False)


DEFAULT_DESERIALIZE_COLUMNS = {
    "face_points_normalized",
    "edge_points_normalized",
    "face_points_world",
    "edge_points_world",
    "face_edge_incidence",
    "edge_face_incidence",
    "face_bbox_world",
    "edge_bbox_world",
    "vertices_world",
    "vertices_unique",
}


class ParquetRowIterable:
    """Stream deserialized rows from one parquet file or a directory of parquet files.

    This keeps the part of the original module that loads parquet data through
    ``pyarrow.dataset`` and deserializes row payloads stored as NumPy byte blobs.
    """

    def __init__(
        self,
        paths: str | Path,
        columns: Sequence[str],
        filter_expr: Optional[pc.Expression] = None,
        deserialize_columns: Optional[Sequence[str]] = None,
        batch_rows_read: int = 4096,
    ) -> None:
        self.paths = str(paths)
        self.columns = list(columns)
        self.filter_expr = filter_expr
        self.deserialize_columns = set(
            deserialize_columns
            if deserialize_columns is not None
            else DEFAULT_DESERIALIZE_COLUMNS
        )
        self.batch_rows_read = int(batch_rows_read)

    def _deserialize_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in row.items():
            if key in self.deserialize_columns and value is not None:
                out[key] = deserialize_array(value)
            else:
                out[key] = value
        return out

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if ds is None:
            raise ImportError("pyarrow is required to stream parquet rows")

        dataset = ds.dataset(self.paths, format="parquet")
        scanner = dataset.scanner(
            columns=self.columns,
            filter=self.filter_expr,
            batch_size=self.batch_rows_read,
        )

        for record_batch in scanner.to_batches():
            cols = {
                name: record_batch.column(i)
                for i, name in enumerate(record_batch.schema.names)
            }
            for row_idx in range(record_batch.num_rows):
                row = {name: column[row_idx].as_py() for name, column in cols.items()}
                yield self._deserialize_row(row)


def load_parquet_rows(
    paths: str | Path,
    columns: Sequence[str],
    filter_expr: Optional[pc.Expression] = None,
    deserialize_columns: Optional[Sequence[str]] = None,
    batch_rows_read: int = 4096,
) -> ParquetRowIterable:
    """Return an iterable over deserialized parquet rows."""
    return ParquetRowIterable(
        paths=paths,
        columns=columns,
        filter_expr=filter_expr,
        deserialize_columns=deserialize_columns,
        batch_rows_read=batch_rows_read,
    )


def get_row_by_index(
    paths: str | Path,
    row_index: int,
    columns: Sequence[str],
    filter_expr: Optional[pc.Expression] = None,
    deserialize_columns: Optional[Sequence[str]] = None,
    batch_rows_read: int = 4096,
) -> Dict[str, Any]:
    """Convenience helper to fetch one deserialized row by global index."""
    if row_index < 0:
        raise ValueError("row_index must be non-negative")

    for idx, row in enumerate(
        load_parquet_rows(
            paths=paths,
            columns=columns,
            filter_expr=filter_expr,
            deserialize_columns=deserialize_columns,
            batch_rows_read=batch_rows_read,
        )
    ):
        if idx == row_index:
            return row

    raise IndexError(f"row_index {row_index} is out of range")


def _resolve_geometry_keys(
    row: Dict[str, Any],
    coordinate_space: str,
) -> Tuple[str, str]:
    if coordinate_space not in {"normalized", "world"}:
        raise ValueError("coordinate_space must be 'normalized' or 'world'")

    face_key = f"face_points_{coordinate_space}"
    edge_key = f"edge_points_{coordinate_space}"

    if face_key not in row:
        raise KeyError(f"Missing required face sample column: {face_key}")
    if edge_key not in row:
        raise KeyError(f"Missing required edge sample column: {edge_key}")

    return face_key, edge_key


def _extract_xyz(samples: np.ndarray) -> np.ndarray:
    arr = np.asarray(samples)
    if arr.shape[-1] < 3:
        raise ValueError(
            f"Expected sample arrays to have at least 3 coordinate channels, got shape {arr.shape}"
        )
    return np.asarray(arr[..., :3], dtype=np.float64)


def _point_key(point: np.ndarray, tol: float) -> Tuple[int, int, int]:
    return tuple(np.round(point / tol).astype(np.int64).tolist())


def _triangle_is_valid(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, area_tol: float
) -> bool:
    if not (
        np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.all(np.isfinite(c))
    ):
        return False
    ab = b - a
    ac = c - a
    return np.linalg.norm(np.cross(ab, ac)) > area_tol


def build_surface_mesh_from_grids(
    face_points: np.ndarray,
    dedup_tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Triangulate per-face UV grids into a global surface mesh.

    Parameters
    ----------
    face_points:
        Expected shape is ``[num_faces, u, v, >=3]``. Only XYZ is used.
    dedup_tol:
        Tolerance used to merge coincident vertices across face grids.

    Returns
    -------
    surface_vertices:
        ``[num_vertices, 3]`` float array.
    surface_faces:
        ``[num_triangles, 3]`` int array with zero-based indices.
    """
    grids = _extract_xyz(face_points)
    if grids.ndim != 4:
        raise ValueError(
            f"Expected face_points to have shape [num_faces, u, v, c], got {grids.shape}"
        )

    vertex_lookup: Dict[Tuple[int, int, int], int] = {}
    vertices: List[np.ndarray] = []
    faces: List[Tuple[int, int, int]] = []
    area_tol = max(dedup_tol * dedup_tol, 1e-16)

    def vertex_index(point: np.ndarray) -> int:
        key = _point_key(point, dedup_tol)
        if key not in vertex_lookup:
            vertex_lookup[key] = len(vertices)
            vertices.append(np.asarray(point, dtype=np.float64))
        return vertex_lookup[key]

    for face_grid in grids:
        u_count, v_count = face_grid.shape[:2]
        local_indices = np.full((u_count, v_count), -1, dtype=np.int64)

        for u in range(u_count):
            for v in range(v_count):
                point = face_grid[u, v]
                if np.all(np.isfinite(point)):
                    local_indices[u, v] = vertex_index(point)

        for u in range(u_count - 1):
            for v in range(v_count - 1):
                a = local_indices[u, v]
                b = local_indices[u + 1, v]
                c = local_indices[u + 1, v + 1]
                d = local_indices[u, v + 1]

                if min(a, b, c) >= 0:
                    pa, pb, pc_ = vertices[a], vertices[b], vertices[c]
                    if _triangle_is_valid(pa, pb, pc_, area_tol):
                        faces.append((a, b, c))

                if min(a, c, d) >= 0:
                    pa, pc_, pd = vertices[a], vertices[c], vertices[d]
                    if _triangle_is_valid(pa, pc_, pd, area_tol):
                        faces.append((a, c, d))

    if vertices:
        return np.vstack(vertices).astype(np.float32), np.asarray(faces, dtype=np.int32)
    return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.int32)


def build_edge_polylines(
    edge_points: np.ndarray,
    dedup_tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert sampled edge curves into global points plus line connectivity.

    Parameters
    ----------
    edge_points:
        Expected shape is ``[num_edges, samples_per_edge, >=3]``. Only XYZ is used.
    dedup_tol:
        Tolerance used to merge coincident points.

    Returns
    -------
    edge_vertices:
        ``[num_points, 3]`` float array.
    edge_segments:
        ``[num_segments, 2]`` int array with zero-based indices.
    """
    curves = _extract_xyz(edge_points)
    if curves.ndim != 3:
        raise ValueError(
            f"Expected edge_points to have shape [num_edges, samples_per_edge, c], got {curves.shape}"
        )

    point_lookup: Dict[Tuple[int, int, int], int] = {}
    points: List[np.ndarray] = []
    segments: List[Tuple[int, int]] = []
    seen_segments: set[Tuple[int, int]] = set()

    def point_index(point: np.ndarray) -> int:
        key = _point_key(point, dedup_tol)
        if key not in point_lookup:
            point_lookup[key] = len(points)
            points.append(np.asarray(point, dtype=np.float64))
        return point_lookup[key]

    for curve in curves:
        curve_indices: List[int] = []
        for point in curve:
            if not np.all(np.isfinite(point)):
                continue
            idx = point_index(point)
            if not curve_indices or idx != curve_indices[-1]:
                curve_indices.append(idx)

        for start, end in zip(curve_indices[:-1], curve_indices[1:]):
            if start == end:
                continue
            seg = (start, end)
            if seg not in seen_segments:
                seen_segments.add(seg)
                segments.append(seg)

    if points:
        return np.vstack(points).astype(np.float32), np.asarray(
            segments, dtype=np.int32
        )
    return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.int32)


def get_surface_and_edge_geometry(
    row: Dict[str, Any],
    coordinate_space: str = "normalized",
    dedup_tol: float = 1e-6,
) -> Dict[str, np.ndarray]:
    """Return triangulated surface geometry and polyline edge geometry for one row."""
    face_key, edge_key = _resolve_geometry_keys(row, coordinate_space)

    surface_vertices, surface_faces = build_surface_mesh_from_grids(
        row[face_key],
        dedup_tol=dedup_tol,
    )
    edge_vertices, edge_segments = build_edge_polylines(
        row[edge_key],
        dedup_tol=dedup_tol,
    )

    return {
        "surface_vertices": surface_vertices,
        "surface_faces": surface_faces,
        "edge_vertices": edge_vertices,
        "edge_segments": edge_segments,
    }


def write_surface_obj(
    vertices: np.ndarray,
    faces: np.ndarray,
    obj_path: str | Path,
) -> Path:
    """Write a triangle mesh OBJ file."""
    path = Path(obj_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        f.write("# Auto-generated surface mesh\n")
        for x, y, z in np.asarray(vertices, dtype=np.float64):
            f.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        for i, j, k in np.asarray(faces, dtype=np.int64):
            f.write(f"f {i + 1} {j + 1} {k + 1}\n")

    return path


def write_edge_obj(
    points: np.ndarray,
    segments: np.ndarray,
    obj_path: str | Path,
) -> Path:
    """Write an edge polyline OBJ file."""
    path = Path(obj_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        f.write("# Auto-generated edge polylines\n")
        for x, y, z in np.asarray(points, dtype=np.float64):
            f.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        for start, end in np.asarray(segments, dtype=np.int64):
            f.write(f"l {start + 1} {end + 1}\n")

    return path


def export_row_to_obj(
    row: Dict[str, Any],
    surface_obj_path: str | Path,
    edge_obj_path: str | Path,
    coordinate_space: str = "normalized",
    dedup_tol: float = 1e-6,
) -> Dict[str, Any]:
    """Build geometry from one row and write surface and edge OBJ files."""
    geometry = get_surface_and_edge_geometry(
        row=row,
        coordinate_space=coordinate_space,
        dedup_tol=dedup_tol,
    )
    write_surface_obj(
        vertices=geometry["surface_vertices"],
        faces=geometry["surface_faces"],
        obj_path=surface_obj_path,
    )
    write_edge_obj(
        points=geometry["edge_vertices"],
        segments=geometry["edge_segments"],
        obj_path=edge_obj_path,
    )
    return geometry


def export_parquet_row_to_obj(
    paths: str | Path,
    row_index: int,
    surface_obj_path: str | Path,
    edge_obj_path: str | Path,
    coordinate_space: str = "normalized",
    filter_expr: Optional[pc.Expression] = None,
    batch_rows_read: int = 4096,
    dedup_tol: float = 1e-6,
) -> Dict[str, Any]:
    """Load one row directly from parquet and export it to two OBJ files."""
    columns = [
        f"face_points_{coordinate_space}",
        f"edge_points_{coordinate_space}",
        "solid_name",
    ]
    row = get_row_by_index(
        paths=paths,
        row_index=row_index,
        columns=columns,
        filter_expr=filter_expr,
        batch_rows_read=batch_rows_read,
    )
    return export_row_to_obj(
        row=row,
        surface_obj_path=surface_obj_path,
        edge_obj_path=edge_obj_path,
        coordinate_space=coordinate_space,
        dedup_tol=dedup_tol,
    )


if __name__ == "__main__":
    OUTPUT = "/home/vthamizharas/Documents/TRELLIS.2/brep_parquet_outputs"

    export_parquet_row_to_obj(
        paths="/home/vthamizharas/Documents/TRELLIS.2/datasets/AutoBrep_Dataset/",
        row_index=0,
        surface_obj_path=f"{OUTPUT}/surface.obj",
        edge_obj_path=f"{OUTPUT}/edge.obj",
    )
