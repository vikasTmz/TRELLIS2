from __future__ import annotations

from dataclasses import dataclass, asdict
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from functools import wraps
from tqdm.auto import tqdm
from typing import Optional, Sequence, Set, Tuple, List, Dict
import hashlib
import json
import time
import random

import numpy as np
import trimesh
from skimage import measure
from scipy.spatial import cKDTree


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


# -------------------------------------------- #
# Helper functions for point sampling
# -------------------------------------------- #


def _filter_min_distance_grid(
    points: np.ndarray, min_dist: float, shuffle: bool = False, seed: int = 0
):
    """
    Greedy min-distance filter using a 3D spatial hash grid.
    Ensures no two returned points are within min_dist of each other.
    """
    if min_dist is None or min_dist <= 0:
        return points

    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] == 0:
        return pts

    # Optionally randomize order to avoid directional bias
    if shuffle:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(pts))
        pts_iter = pts[order]
    else:
        pts_iter = pts

    cell = float(min_dist)
    inv_cell = 1.0 / cell
    min_dist2 = cell * cell

    # grid maps (ix,iy,iz) -> list of kept points (as np arrays)
    grid = {}
    kept = []

    for p in pts_iter:
        key = tuple(np.floor(p * inv_cell).astype(np.int32))

        # Gather candidates from this cell and its neighbors
        candidates = []
        kx, ky, kz = key
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    nk = (kx + dx, ky + dy, kz + dz)
                    if nk in grid:
                        candidates.extend(grid[nk])

        # Check actual distance only against nearby candidates
        if candidates:
            c = np.asarray(candidates, dtype=np.float32)  # (M,3)
            d2 = np.sum((c - p) ** 2, axis=1)
            if np.any(d2 < min_dist2):
                continue  # too close to an existing kept point

        # Keep it
        kept.append(p)
        grid.setdefault(key, []).append(p)

    kept = np.asarray(kept, dtype=np.float32)

    # If we shuffled, order doesn't matter usually. If you want original ordering preserved,
    # don't shuffle. Otherwise you can sort kept by something if needed.
    return kept


def _filter_min_distance_grid_with_indices(points, min_dist, shuffle=False, seed=0):
    """
    Grid-based min-distance filter.

    Returns:
        filtered_points:
            Kept points.

        kept_indices:
            Indices of kept points in the original input array.
    """
    points = np.asarray(points, dtype=np.float32)

    if len(points) == 0:
        return points, np.array([], dtype=np.int64)

    if shuffle:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(points))
    else:
        order = np.arange(len(points))

    cell_size = float(min_dist)
    inv_cell_size = 1.0 / cell_size
    min_dist_sq = min_dist * min_dist

    grid = {}
    kept_indices = []

    neighbor_offsets = np.array(
        [[dx, dy, dz] for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
        dtype=np.int32,
    )

    for idx in order:
        p = points[idx]
        cell = tuple(np.floor(p * inv_cell_size).astype(np.int32))

        keep = True

        for offset in neighbor_offsets:
            neighbor_cell = tuple(np.array(cell, dtype=np.int32) + offset)

            if neighbor_cell not in grid:
                continue

            for kept_idx in grid[neighbor_cell]:
                q = points[kept_idx]

                if np.sum((p - q) ** 2) < min_dist_sq:
                    keep = False
                    break

            if not keep:
                break

        if keep:
            kept_indices.append(idx)
            grid.setdefault(cell, []).append(idx)

    kept_indices = np.array(kept_indices, dtype=np.int64)

    # Preserve original point order in the output.
    kept_indices = np.sort(kept_indices)

    return points[kept_indices], kept_indices


def sample_and_export_boundary_points(
    boundary_segments,
    points_per_unit=100,
    min_dist=None,  # <-- hyperparameter "x" (same units as your points)
    shuffle_filter=False,  # optional: more uniform selection
    seed=0,
):
    """
    Samples points along line segments proportionally to length and exports to PLY.

    Args:
        boundary_segments: List/array of shape (N, 2, 3) or list of [p1, p2].
        output_path: File path for the .ply file.
        points_per_unit: Density of points (points per unit of length).
        min_dist: If set, enforces a minimum distance between all sampled points.
        shuffle_filter: If True, randomizes point order before filtering (often nicer).
        seed: RNG seed used when shuffle_filter=True.
    """
    all_points = []

    print(f"Sampling points with density: {points_per_unit} pts/unit...")

    for p1, p2 in boundary_segments:
        p1, p2 = np.array(p1, dtype=np.float32), np.array(p2, dtype=np.float32)

        segment_vector = p2 - p1
        segment_length = np.linalg.norm(segment_vector)

        if segment_length < 1e-8:
            continue

        num_samples = max(2, int(segment_length * points_per_unit))
        samples = np.linspace(p1, p2, num_samples, dtype=np.float32)
        all_points.append(samples)

    if not all_points:
        print("No points sampled. Check your boundary_segments input.")
        return None

    final_points = np.vstack(all_points)

    # --- Enforce minimum separation ---
    if min_dist is not None and min_dist > 0:
        before = len(final_points)
        final_points = _filter_min_distance_grid(
            final_points, min_dist, shuffle=shuffle_filter, seed=seed
        )
        after = len(final_points)
        print(f"Min-dist filtering (x={min_dist}): {before} -> {after} points")

    return final_points


def sample_and_export_boundary_points_new(
    boundary_segments,
    points_per_unit=100,
    min_dist=None,
    shuffle_filter=False,
    seed=0,
):
    """
    Samples points along line segments and also returns explicit segment endpoints.

    Args:
        boundary_segments:
            List/array of shape (K, 2, 3), where each item is [p1, p2].

        points_per_unit:
            Density of points per unit length.

        min_dist:
            If set, enforces a minimum distance between all sampled points.

        shuffle_filter:
            If True, randomizes point order before min-distance filtering.

        seed:
            RNG seed used when shuffle_filter=True.

    Returns:
        final_points:
            Array of shape (N, 3).

        final_segments:
            Array of shape (M, 2, 3), where each segment is explicitly
            represented as [[x1, y1, z1], [x2, y2, z2]].
    """
    boundary_segments = np.asarray(boundary_segments, dtype=np.float32)

    if boundary_segments.ndim != 3 or boundary_segments.shape[1:] != (2, 3):
        raise ValueError("boundary_segments must have shape (K, 2, 3)")

    all_points = []
    all_edge_ids = []
    all_t_values = []

    print(f"Sampling points with density: {points_per_unit} pts/unit...")

    for edge_id, (p1, p2) in enumerate(boundary_segments):
        p1 = np.asarray(p1, dtype=np.float32)
        p2 = np.asarray(p2, dtype=np.float32)

        segment_vector = p2 - p1
        segment_length = np.linalg.norm(segment_vector)

        if segment_length < 1e-8:
            continue

        num_samples = max(2, int(segment_length * points_per_unit))

        t_values = np.linspace(0.0, 1.0, num_samples, dtype=np.float32)
        samples = (
            p1[None, :] * (1.0 - t_values[:, None]) + p2[None, :] * t_values[:, None]
        ).astype(np.float32)

        all_points.append(samples)
        all_edge_ids.append(np.full(num_samples, edge_id, dtype=np.int64))
        all_t_values.append(t_values)

    if not all_points:
        print("No points sampled. Check your boundary_segments input.")
        return None, None

    candidate_points = np.vstack(all_points)
    candidate_edge_ids = np.concatenate(all_edge_ids)
    candidate_t_values = np.concatenate(all_t_values)

    kept_candidate_indices = np.arange(len(candidate_points), dtype=np.int64)

    # --- Enforce minimum separation ---
    if min_dist is not None and min_dist > 0:
        before = len(candidate_points)

        final_points, kept_candidate_indices = _filter_min_distance_grid_with_indices(
            candidate_points,
            min_dist,
            shuffle=shuffle_filter,
            seed=seed,
        )

        after = len(final_points)
        print(f"Min-dist filtering (x={min_dist}): {before} -> {after} points")
    else:
        final_points = candidate_points

    kept_edge_ids = candidate_edge_ids[kept_candidate_indices]
    kept_t_values = candidate_t_values[kept_candidate_indices]

    # Rebuild explicit segments between consecutive kept points
    # that lie on the same original boundary segment.
    final_segments = []

    for edge_id in range(len(boundary_segments)):
        local_indices = np.where(kept_edge_ids == edge_id)[0]

        if len(local_indices) < 2:
            continue

        # Sort along the original edge direction.
        local_indices = local_indices[np.argsort(kept_t_values[local_indices])]

        local_points = final_points[local_indices]

        edge_segments = np.stack(
            [local_points[:-1], local_points[1:]],
            axis=1,
        )

        final_segments.append(edge_segments)

    if final_segments:
        final_segments = np.vstack(final_segments).astype(np.float32)
    else:
        final_segments = np.empty((0, 2, 3), dtype=np.float32)

    return final_points.astype(np.float32), final_segments


def normalize_point_cloud_list(points):
    points = np.asarray(points, dtype=np.float32)
    # Centering
    centroid = np.mean(points.reshape(-1, 3), axis=0)
    points -= centroid

    # Scaling to unit sphere
    furthest_distance = np.max(np.sqrt(np.sum(points.reshape(-1, 3) ** 2, axis=1)))
    points /= furthest_distance

    return points, centroid, furthest_distance


def normalize_point_cloud(points):
    # Centering
    centroid = np.mean(points, axis=0)
    points -= centroid

    # Scaling to unit sphere
    furthest_distance = np.max(np.sqrt(np.sum(points**2, axis=1)))
    points /= furthest_distance

    return points, centroid, furthest_distance


def unnormalize_point_cloud(normalized_points, centroid, furthest_distance):
    normalized_points = np.asarray(normalized_points, dtype=np.float32)

    points = normalized_points * furthest_distance
    points = points + centroid

    return points


# ---------------------------------------------------------------- #
# Helper functions for metaball/smooth-union surface construction
# ---------------------------------------------------------------- #


def estimate_point_weights(points, knn=8, density_dim=1, eps=1e-6, clip=(0.25, 4.0)):
    """
    Estimate inverse-density weights for each point.

    density_dim:
        1 -> curve-like sampling
        2 -> surface-like sampling
        3 -> volumetric sampling
    """
    pts = np.asarray(points, dtype=np.float32)
    tree = cKDTree(pts)

    k = min(knn + 1, len(pts))
    dists, _ = tree.query(pts, k=k)  # includes self at column 0

    if k <= 1:
        return np.ones(len(pts), dtype=np.float32)

    hk = dists[:, -1]  # distance to kth neighbor
    w = np.maximum(hk, eps) ** density_dim

    # normalize so a "typical" point has weight ~1
    med = np.median(w)
    if med > 0:
        w = w / med

    # optional: prevent extreme outliers from dominating
    w = np.clip(w, clip[0], clip[1])

    return w.astype(np.float32)


def build_smooth_union_sdf_new(points, xs, ys, zs, radius, k, weights=None, eps=1e-6):
    band = -np.log(eps) / k
    R = radius + band

    dtype = np.float32

    nx, ny, nz = len(xs), len(ys), len(zs)

    m = np.full((nx, ny, nz), dtype(1e6), dtype=dtype)
    s = np.zeros((nx, ny, nz), dtype=dtype)

    xs_f = xs.astype(dtype, copy=False)
    ys_f = ys.astype(dtype, copy=False)
    zs_f = zs.astype(dtype, copy=False)

    radius = dtype(radius)
    k = dtype(k)

    if weights is None:
        weights = np.ones(len(points), dtype=dtype)
    else:
        weights = np.asarray(weights, dtype=dtype)

    pts = points.astype(dtype, copy=False)

    for (px, py, pz), w in tqdm(
        zip(pts, weights),
        total=len(pts),
        desc="Blending spheres",
        unit="sphere",
        bar_format="{l_bar}{bar} {n_fmt}/{total_fmt} "
        "[elapsed: {elapsed} | remaining: {remaining} | {rate_fmt}{postfix}]",
    ):
        ix0 = np.searchsorted(xs_f, px - R, side="left")
        ix1 = np.searchsorted(xs_f, px + R, side="right")
        iy0 = np.searchsorted(ys_f, py - R, side="left")
        iy1 = np.searchsorted(ys_f, py + R, side="right")
        iz0 = np.searchsorted(zs_f, pz - R, side="left")
        iz1 = np.searchsorted(zs_f, pz + R, side="right")

        if ix0 >= ix1 or iy0 >= iy1 or iz0 >= iz1:
            continue

        dx2 = (xs_f[ix0:ix1] - px) ** 2
        dy2 = (ys_f[iy0:iy1] - py) ** 2
        dz2 = (zs_f[iz0:iz1] - pz) ** 2

        di = (
            np.sqrt(dx2[:, None, None] + dy2[None, :, None] + dz2[None, None, :])
            - radius
        )

        m0 = m[ix0:ix1, iy0:iy1, iz0:iz1]
        s0 = s[ix0:ix1, iy0:iy1, iz0:iz1]

        new_m = np.minimum(m0, di)

        scale = np.exp(-k * (m0 - new_m)).astype(dtype, copy=False)
        contrib = (w * np.exp(-k * (di - new_m))).astype(dtype, copy=False)

        s_new = s0 * scale + contrib

        m0[...] = new_m
        s0[...] = s_new

    out = m.copy()
    mask = s > 0
    out[mask] = m[mask] - (1.0 / k) * np.log(s[mask])

    return out


def points_to_metaball_mesh_new(
    points: np.ndarray,
    radius: float = 0.2,
    smooth_k: float | None = None,
    voxel_size: float = 0.02,
    padding: float = 0.4,
    eps: float = 1e-6,
) -> trimesh.Trimesh:
    print("Convert 3D points to a blended metaball/smooth-union surface mesh...")

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be shape (N, 3)")
    if len(points) == 0:
        raise ValueError("points array is empty")

    if smooth_k is None:
        smooth_k = 8.0 / max(radius, eps)

    # NEW: density compensation
    weights = estimate_point_weights(points, knn=8, density_dim=1)

    pmin = points.min(axis=0) - (padding + 2.0 * radius)
    pmax = points.max(axis=0) + (padding + 2.0 * radius)

    dims = np.ceil((pmax - pmin) / voxel_size).astype(int) + 1
    nx, ny, nz = dims.tolist()

    xs = np.linspace(pmin[0], pmax[0], nx, dtype=np.float32)
    ys = np.linspace(pmin[1], pmax[1], ny, dtype=np.float32)
    zs = np.linspace(pmin[2], pmax[2], nz, dtype=np.float32)

    print("Build smooth-union SDF over the grid")
    print(f"Processing {points.shape[0]} points")

    d = build_smooth_union_sdf_new(
        points,
        xs,
        ys,
        zs,
        radius=radius,
        k=smooth_k,
        weights=weights,
        eps=eps,
    )

    dx = xs[1] - xs[0] if nx > 1 else 1.0
    dy = ys[1] - ys[0] if ny > 1 else 1.0
    dz = zs[1] - zs[0] if nz > 1 else 1.0

    print("Running marching cubes...")

    verts, faces, norms, _ = measure.marching_cubes(
        volume=d,
        level=0.0,
        spacing=(dx, dy, dz),
    )

    verts = verts + pmin

    mesh = trimesh.Trimesh(
        vertices=verts, faces=faces, vertex_normals=norms, process=True
    )

    return mesh


def estimate_point_convolution_weights(
    points: np.ndarray,
    knn: int = 8,
    density_dim: int = 1,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Density-compensation weights for an unordered sampled skeleton.

    density_dim=1: points sampled along curves/skeletons.
    density_dim=2: points sampled on surface patches.
    density_dim=3: points sampled in volumes.
    """
    points = np.asarray(points, dtype=np.float32)
    n = len(points)
    if n == 1:
        return np.ones(1, dtype=np.float32)

    k = min(knn + 1, n)
    dists, _ = cKDTree(points).query(points, k=k)

    if k == 2:
        local_spacing = dists[:, 1]
    else:
        local_spacing = np.mean(dists[:, 1:], axis=1)

    local_spacing = np.maximum(local_spacing, eps)
    med = float(np.median(local_spacing))
    if med > eps:
        local_spacing = np.clip(local_spacing, 0.25 * med, 4.0 * med)

    return (local_spacing**density_dim).astype(np.float32)


def _poly6_kernel_from_r2(r2: np.ndarray, support_radius: float) -> np.ndarray:
    """
    Compact radial convolution kernel:

        K(r) = (1 - (r / R)^2)^3,  r < R
             = 0,                  otherwise

    where R = support_radius.
    """
    u = r2 / (support_radius * support_radius)
    return np.where(u < 1.0, (1.0 - u) ** 3, 0.0).astype(np.float32, copy=False)


def _line_iso_level_for_poly6(
    radius: float,
    support_radius: float,
    n_quad: int = 96,
) -> float:
    """
    Iso-level that makes an isolated infinite line skeleton have approximately
    the requested surface radius when using the poly6 kernel and arc-length
    weights.

    F(d) = integral K(sqrt(d^2 + t^2)) dt
    """
    radius = float(radius)
    support_radius = float(support_radius)

    if not (0.0 < radius < support_radius):
        raise ValueError(
            "For line iso calibration, require 0 < radius < support_radius"
        )

    half_len = np.sqrt(support_radius * support_radius - radius * radius)
    x, w = np.polynomial.legendre.leggauss(n_quad)

    t = half_len * x
    r2 = radius * radius + t * t
    vals = _poly6_kernel_from_r2(r2, support_radius).astype(np.float64)

    return float(half_len * np.sum(w * vals))


def _point_iso_level_for_poly6(radius: float, support_radius: float) -> float:
    """
    Iso-level for a single point primitive with unit weight.
    Useful only when you intentionally want point-blob behavior.
    """
    if not (0.0 < radius < support_radius):
        raise ValueError(
            "For point iso calibration, require 0 < radius < support_radius"
        )

    u = (radius / support_radius) ** 2
    return float((1.0 - u) ** 3)


def _segments_to_samples(
    points: np.ndarray,
    segments: np.ndarray,
    sample_step: float,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert line segments to midpoint samples with arc-length weights.

    segments can be either:
      - integer array of shape (M, 2), indexing into points
      - float array of shape (M, 2, 3), explicit segment endpoints
    """
    points = np.asarray(points, dtype=np.float32)
    segments = np.asarray(segments)

    if (
        segments.ndim == 2
        and segments.shape[1] == 2
        and np.issubdtype(segments.dtype, np.integer)
    ):
        p0 = points[segments[:, 0]]
        p1 = points[segments[:, 1]]
    elif segments.ndim == 3 and segments.shape[1:] == (2, 3):
        p0 = segments[:, 0, :].astype(np.float32, copy=False)
        p1 = segments[:, 1, :].astype(np.float32, copy=False)
    else:
        raise ValueError(
            "segments must have shape (M, 2) integer indices or (M, 2, 3) endpoints"
        )

    sample_step = float(sample_step)
    if sample_step <= 0:
        raise ValueError("sample_step must be positive")

    all_samples = []
    all_weights = []

    for a, b in zip(p0, p1):
        v = b - a
        length = float(np.linalg.norm(v))
        if length <= eps:
            continue

        n = max(1, int(np.ceil(length / sample_step)))
        t = (np.arange(n, dtype=np.float32) + 0.5) / n

        samples = a[None, :] * (1.0 - t[:, None]) + b[None, :] * t[:, None]
        weights = np.full(n, length / n, dtype=np.float32)

        all_samples.append(samples)
        all_weights.append(weights)

    if not all_samples:
        raise ValueError("segments produced no valid samples")

    return (
        np.vstack(all_samples).astype(np.float32),
        np.concatenate(all_weights).astype(np.float32),
    )


def build_convolution_field(
    samples: np.ndarray,
    weights: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    support_radius: float,
    progress: bool = True,
) -> np.ndarray:
    """
    Build scalar convolution field:

        F(x) = sum_i w_i K(||x - s_i||)

    This uses local kernel splatting, so it only touches voxels inside the
    compact support radius around each skeleton sample.
    """
    samples = np.asarray(samples, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)

    nx, ny, nz = len(xs), len(ys), len(zs)
    field = np.zeros((nx, ny, nz), dtype=np.float32)

    R = float(support_radius)
    R2 = R * R

    for i, (p, w) in enumerate(zip(samples, weights)):
        ix0 = max(0, int(np.searchsorted(xs, p[0] - R, side="left")))
        ix1 = min(nx, int(np.searchsorted(xs, p[0] + R, side="right")))

        iy0 = max(0, int(np.searchsorted(ys, p[1] - R, side="left")))
        iy1 = min(ny, int(np.searchsorted(ys, p[1] + R, side="right")))

        iz0 = max(0, int(np.searchsorted(zs, p[2] - R, side="left")))
        iz1 = min(nz, int(np.searchsorted(zs, p[2] + R, side="right")))

        if ix0 >= ix1 or iy0 >= iy1 or iz0 >= iz1:
            continue

        dx2 = (xs[ix0:ix1] - p[0])[:, None, None] ** 2
        dy2 = (ys[iy0:iy1] - p[1])[None, :, None] ** 2
        dz2 = (zs[iz0:iz1] - p[2])[None, None, :] ** 2

        r2 = dx2 + dy2 + dz2
        u = r2 / R2

        contrib = np.where(u < 1.0, (1.0 - u) ** 3, 0.0).astype(
            np.float32,
            copy=False,
        )

        field[ix0:ix1, iy0:iy1, iz0:iz1] += np.float32(w) * contrib

        if progress and (i + 1) % 1000 == 0:
            print(f"  splatted {i + 1}/{len(samples)} convolution samples")

    return field


def points_to_convolution_surface_mesh(
    points: np.ndarray,
    radius: float = 0.2,
    support_radius: float | None = None,
    voxel_size: float = 0.02,
    padding: float = 0.4,
    segments: np.ndarray | None = None,
    sample_step: float | None = None,
    iso_level: float | None = None,
    iso_mode: str = "line",
    knn: int = 8,
    density_dim: int = 1,
    max_grid_cells: int = 220_000_000,
    process: bool = True,
) -> trimesh.Trimesh:
    """
    Convert a sampled skeleton / point cloud to a convolution-surface mesh.

    Parameters
    ----------
    points:
        (N, 3) point cloud. For best results these are samples on a skeleton.

    radius:
        Desired visible surface radius around an isolated curve skeleton.

    support_radius:
        Kernel cutoff radius. Smaller values reduce unwanted interaction between
        nearby branches. Must be larger than radius. Default: 2 * radius.

    voxel_size:
        Marching-cubes grid spacing.

    padding:
        Extra domain padding beyond the kernel support.

    segments:
        Optional line skeleton. Either:
          - (M, 2) integer indices into points, or
          - (M, 2, 3) explicit segment endpoints.

        This is strongly recommended when points are ordered samples of curves.

    sample_step:
        Segment integration sample spacing. Default: voxel_size.

    iso_level:
        Marching-cubes level. If None, it is calibrated from radius/support_radius.

    iso_mode:
        "line" for curve skeletons.
        "point" for isolated point-blob behavior.

    knn, density_dim:
        Used only when segments is None.
        density_dim=1 is appropriate for point samples along curves.
    """
    print("Convert 3D points to a convolution surface mesh...")

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be shape (N, 3)")
    if len(points) == 0:
        raise ValueError("points array is empty")

    radius = float(radius)
    voxel_size = float(voxel_size)
    padding = float(padding)

    if radius <= 0:
        raise ValueError("radius must be positive")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")

    if support_radius is None:
        support_radius = 2.0 * radius

    support_radius = float(support_radius)
    if not (radius < support_radius):
        raise ValueError("support_radius must be larger than radius")

    if sample_step is None:
        sample_step = voxel_size

    if segments is not None:
        samples, weights = _segments_to_samples(points, segments, sample_step)
        print(
            f"Using {len(samples)} line-integration samples from {len(segments)} segments"
        )
    else:
        samples = points
        weights = estimate_point_convolution_weights(
            points,
            knn=knn,
            density_dim=density_dim,
        )
        print(f"Using {len(samples)} weighted point samples")

    if iso_level is None:
        if iso_mode == "line":
            iso_level = _line_iso_level_for_poly6(radius, support_radius)
        elif iso_mode == "point":
            iso_level = _point_iso_level_for_poly6(radius, support_radius)
        else:
            raise ValueError('iso_mode must be "line" or "point"')

    iso_level = float(iso_level)
    print(f"support_radius={support_radius:g}, iso_level={iso_level:g}")

    pmin = points.min(axis=0) - (padding + support_radius)
    pmax = points.max(axis=0) + (padding + support_radius)

    dims = np.ceil((pmax - pmin) / voxel_size).astype(int) + 1
    nx, ny, nz = dims.tolist()

    grid_cells = int(nx) * int(ny) * int(nz)
    if grid_cells > max_grid_cells:
        raise MemoryError(
            f"Grid has {grid_cells:,} cells ({nx} x {ny} x {nz}). "
            "Increase voxel_size, reduce padding/support_radius, or raise max_grid_cells."
        )

    xs = (pmin[0] + np.arange(nx, dtype=np.float32) * voxel_size).astype(np.float32)
    ys = (pmin[1] + np.arange(ny, dtype=np.float32) * voxel_size).astype(np.float32)
    zs = (pmin[2] + np.arange(nz, dtype=np.float32) * voxel_size).astype(np.float32)

    print(f"Building convolution field over grid {nx} x {ny} x {nz}")

    field = build_convolution_field(
        samples=samples,
        weights=weights,
        xs=xs,
        ys=ys,
        zs=zs,
        support_radius=support_radius,
    )

    fmin = float(field.min())
    fmax = float(field.max())

    if not (fmin <= iso_level <= fmax):
        raise ValueError(
            f"iso_level={iso_level:g} is outside field range [{fmin:g}, {fmax:g}]. "
            "Try lowering iso_level, increasing support_radius, or using denser skeleton samples."
        )

    print("Running marching cubes...")

    verts, faces, norms, _ = measure.marching_cubes(
        volume=field,
        level=iso_level,
        spacing=(voxel_size, voxel_size, voxel_size),
    )

    verts = verts + pmin[None, :]

    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        vertex_normals=norms,
        process=process,
    )

    if process:
        mesh.fix_normals()

    return mesh


# ---------------------------------------------------------------- #
# Helper functions for random obj utils
# ---------------------------------------------------------------- #


def export_mesh(mesh: trimesh.Trimesh, out_ply: str, out_glb: str) -> None:
    if out_ply is not None:
        mesh.export(out_ply)  # PLY
    # GLB export: wrap in a scene for best compatibility
    scene = trimesh.Scene(mesh)
    scene.export(out_glb)


def analyze_obj(path: Path):
    vertices = 0
    texcoords = 0
    normals = 0
    faces = 0
    triangles = 0
    objects = 0
    groups = 0

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if line.startswith("v "):
                vertices += 1
            elif line.startswith("vt "):
                texcoords += 1
            elif line.startswith("vn "):
                normals += 1
            elif line.startswith("f "):
                faces += 1

                # OBJ faces can have 3, 4, or more vertices.
                # An n-gon becomes n - 2 triangles when triangulated.
                face_vertices = line.split()[1:]
                if len(face_vertices) >= 3:
                    triangles += len(face_vertices) - 2

            elif line.startswith("o "):
                objects += 1
            elif line.startswith("g "):
                groups += 1

    file_size = path.stat().st_size

    complexity_score = (
        triangles
        + 0.25 * vertices
        + 0.1 * texcoords
        + 0.1 * normals
        + 5 * objects
        + 2 * groups
    )

    return {
        "file": str(path),
        "size_bytes": file_size,
        "vertices": vertices,
        "texcoords": texcoords,
        "normals": normals,
        "faces": faces,
        "triangles": triangles,
        "objects": objects,
        "groups": groups,
        "score": complexity_score,
    }


def sort_objs(obj_list, cache_path=None):
    results = []
    print(f" Sorting {len(obj_list)} OBJ files by complexity...")

    # check if file exists
    if cache_path.exists():
        print(f"  Loading cached complexity results from {cache_path}...")
        with cache_path.open("r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        for obj_path in obj_list:
            results.append(analyze_obj(Path(obj_path)))
        if cache_path is not None:
            with cache_path.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

    results.sort(
        key=lambda x: (
            x["triangles"],
            x["vertices"],
            x["size_bytes"],
        ),
        reverse=True,
    )

    return results


# ---------------------------------------------------------------- #
# Helper functions for Generalized Voronoi Diagram construction
# ---------------------------------------------------------------- #

# Version 1
# Version 2

Vec3 = np.ndarray
GroupId = int
Pair = Tuple[int, int]


@dataclass
class PlanePrimitive:
    group_id: int
    normal: List[float]
    offset: float
    point: List[float]
    rms_error: float
    max_abs_error: float
    area: float


@dataclass
class PlanarGVDSheet:
    """
    One analytic GVD sheet primitive.

    The sheet is the polygonal support of an analytic plane:
        normal · x + offset = 0

    pair:
        The two B-Rep face group IDs whose Voronoi boundary this sheet represents.

    branch:
        "minus" means phi_i - phi_j = 0.
        "plus" means phi_i + phi_j = 0.

    polygon:
        Vertices of the clipped sheet polygon.
    """

    pair: Pair
    branch: str
    normal: List[float]
    offset: float
    polygon: List[List[float]]
    source_boundary_count: int


def fit_planes_from_groups(
    vertices: Sequence[Sequence[float]],
    groups: Dict[int, dict],
    *,
    coplanarity_warning_ratio: float = 1e-4,
) -> Dict[int, PlanePrimitive]:
    """
    Fits one plane primitive per OBJ/B-Rep face group.

    This uses an area-weighted PCA fit over each group's triangles.
    It is robust to arbitrary triangle orientation.

    Returns:
        planes[group_id] = PlanePrimitive(...)
    """
    V = np.asarray(vertices, dtype=np.float64)
    planes: Dict[int, PlanePrimitive] = {}

    bbox_diag = np.linalg.norm(V.max(axis=0) - V.min(axis=0))
    if bbox_diag <= 0:
        raise ValueError("Degenerate input vertices.")

    for gid, data in sorted(groups.items()):
        face_records = data.get("faces", [])
        if not face_records:
            continue

        tris = []
        tri_areas = []

        for rec in face_records:
            idx = rec["indices"] if isinstance(rec, dict) else rec
            p0, p1, p2 = V[np.asarray(idx, dtype=np.int64)]
            area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
            if area <= 1e-20:
                continue
            tris.append([p0, p1, p2])
            tri_areas.append(area)

        if not tris:
            continue

        tris = np.asarray(tris, dtype=np.float64)  # (T, 3, 3)
        tri_areas = np.asarray(tri_areas, dtype=np.float64)

        centers = tris.mean(axis=1)
        total_area = float(tri_areas.sum())
        centroid = (centers * tri_areas[:, None]).sum(axis=0) / total_area

        # Area-weighted covariance over triangle vertices.
        X = tris.reshape(-1, 3) - centroid[None, :]
        W = np.repeat(tri_areas / 3.0, 3)
        C = (X * W[:, None]).T @ X / W.sum()

        evals, evecs = np.linalg.eigh(C)
        n = evecs[:, np.argmin(evals)]
        n = _normalize(n)

        # Try to orient the fitted normal consistently with the mesh normals.
        area_normal = np.zeros(3, dtype=np.float64)
        for tri in tris:
            p0, p1, p2 = tri
            area_normal += np.cross(p1 - p0, p2 - p0)
        if np.linalg.norm(area_normal) > 1e-12 and np.dot(n, area_normal) < 0:
            n = -n

        d = -float(np.dot(n, centroid))

        all_pts = tris.reshape(-1, 3)
        signed_errors = all_pts @ n + d
        rms_error = float(np.sqrt(np.mean(signed_errors**2)))
        max_abs_error = float(np.max(np.abs(signed_errors)))

        if max_abs_error > coplanarity_warning_ratio * bbox_diag:
            print(
                f"[warning] group {gid} is not very planar: "
                f"max_abs_error={max_abs_error:.6g}, "
                f"bbox_diag={bbox_diag:.6g}. "
                "The exported GVD sheet is a planar approximation."
            )

        planes[int(gid)] = PlanePrimitive(
            group_id=int(gid),
            normal=n.tolist(),
            offset=d,
            point=centroid.tolist(),
            rms_error=rms_error,
            max_abs_error=max_abs_error,
            area=total_area,
        )

    return planes


def infer_boundary_records_from_groups(
    vertices: Sequence[Sequence[float]],
    groups: Dict[int, dict],
    *,
    ndigits: int = 6,
) -> List[dict]:
    """
    Reconstructs labeled B-Rep boundary records from the same idea as your importer:
    triangle edges are matched by quantized geometric endpoint positions.

    Returns records like:
        {
            "groups": [gid_a, gid_b],
            "points": [[x0, y0, z0], [x1, y1, z1]]
        }

    Use these records instead of plain boundary_segments, because the GVD sheet
    must know which two face groups generated it.
    """
    V = np.asarray(vertices, dtype=np.float64)

    edge_tracker = defaultdict(lambda: {"groups": set(), "points": None})

    def qpoint(p):
        return tuple(round(float(c), ndigits) for c in p)

    def edge_key(a, b):
        qa = qpoint(V[a])
        qb = qpoint(V[b])
        return tuple(sorted((qa, qb)))

    for gid, data in groups.items():
        for rec in data.get("faces", []):
            tri = rec["indices"] if isinstance(rec, dict) else rec
            tri = list(map(int, tri))

            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                key = edge_key(a, b)
                edge_tracker[key]["groups"].add(int(gid))
                if edge_tracker[key]["points"] is None:
                    edge_tracker[key]["points"] = [V[a].tolist(), V[b].tolist()]

    records = []

    for _, info in edge_tracker.items():
        gids = sorted(info["groups"])
        if len(gids) <= 1:
            continue

        p0, p1 = info["points"]

        # Usually there are exactly two groups, but combinations handles
        # non-manifold shared edges too.
        for ga, gb in combinations(gids, 2):
            records.append(
                {
                    "groups": [int(ga), int(gb)],
                    "points": [p0, p1],
                }
            )

    return records


def build_planar_gvd_sheets(
    vertices: Sequence[Sequence[float]],
    groups: Dict[int, dict],
    *,
    boundary_records: Optional[List[dict]] = None,
    mode: str = "bbox",
    branches: str = "both",
    padding: float = 0.08,
    strip_half_width: Optional[float] = None,
    ndigits: int = 6,
) -> Tuple[Dict[int, PlanePrimitive], List[PlanarGVDSheet]]:
    """
    Builds analytic planar GVD sheet primitives.

    Parameters
    ----------
    vertices:
        OBJ vertices.

    groups:
        Your importer's group dictionary.

    boundary_records:
        Optional labeled boundary records. If None, they are inferred from
        groups using geometric duplicate-edge matching.

    mode:
        "bbox":
            For each adjacent B-Rep face pair, export the angle-bisector plane
            clipped by the padded bounding box. This gives large clean sheet
            primitives.

        "edge_strips":
            For each B-Rep boundary segment, export a rectangular strip lying
            in the angle-bisector plane. This gives local thin sheets around
            topological edges and avoids huge global planes.

    branches:
        "both":
            Export both unsigned-distance angle-bisectors:
            phi_i - phi_j = 0 and phi_i + phi_j = 0.

        "minus":
            Export only phi_i - phi_j = 0.

        "plus":
            Export only phi_i + phi_j = 0.

    padding:
        Bounding-box padding fraction for mode="bbox".

    strip_half_width:
        Half-width of rectangular strips for mode="edge_strips".
        If None, uses 10% of the model bounding-box diagonal.

    Returns
    -------
    planes:
        Fitted source plane primitive per B-Rep face group.

    sheets:
        Analytic GVD sheet primitives.
    """
    if mode not in {"bbox", "edge_strips"}:
        raise ValueError("mode must be either 'bbox' or 'edge_strips'.")
    if branches not in {"both", "minus", "plus"}:
        raise ValueError("branches must be 'both', 'minus', or 'plus'.")

    V = np.asarray(vertices, dtype=np.float64)

    planes = fit_planes_from_groups(V, groups)

    if boundary_records is None:
        boundary_records = infer_boundary_records_from_groups(
            V,
            groups,
            ndigits=ndigits,
        )

    pair_to_records: Dict[Pair, List[dict]] = defaultdict(list)
    for rec in boundary_records:
        ga, gb = map(int, rec["groups"])
        pair = _canonical_pair(ga, gb)
        pair_to_records[pair].append(rec)

    bbox_min = V.min(axis=0)
    bbox_max = V.max(axis=0)
    diag = np.linalg.norm(bbox_max - bbox_min)
    pad = float(padding) * diag
    bbox_min = bbox_min - pad
    bbox_max = bbox_max + pad

    if strip_half_width is None:
        strip_half_width = 0.10 * diag

    sheets: List[PlanarGVDSheet] = []

    for pair, records in sorted(pair_to_records.items()):
        ga, gb = pair

        if ga not in planes or gb not in planes:
            continue

        pi = planes[ga]
        pj = planes[gb]

        candidate_bisectors = _bisector_planes(pi, pj, branches=branches)

        for branch_name, n, d in candidate_bisectors:
            if mode == "bbox":
                polygon = clip_plane_to_aabb(n, d, bbox_min, bbox_max)
                if len(polygon) < 3:
                    continue

                sheets.append(
                    PlanarGVDSheet(
                        pair=pair,
                        branch=branch_name,
                        normal=n.tolist(),
                        offset=float(d),
                        polygon=[p.tolist() for p in polygon],
                        source_boundary_count=len(records),
                    )
                )

            else:
                # Local rectangular strips around every B-Rep boundary segment.
                for rec in records:
                    p0, p1 = np.asarray(rec["points"], dtype=np.float64)

                    polygon = make_edge_bisector_strip(
                        p0,
                        p1,
                        n,
                        d,
                        half_width=float(strip_half_width),
                    )

                    if polygon is None:
                        continue

                    sheets.append(
                        PlanarGVDSheet(
                            pair=pair,
                            branch=branch_name,
                            normal=n.tolist(),
                            offset=float(d),
                            polygon=[p.tolist() for p in polygon],
                            source_boundary_count=1,
                        )
                    )

    return planes, sheets


def export_planar_gvd(
    planes: Dict[int, PlanePrimitive],
    sheets: List[PlanarGVDSheet],
    out_prefix: str | Path,
) -> Dict[str, str]:
    """
    Exports analytic GVD data.

    Writes:
        <prefix>_source_planes.json
        <prefix>_gvd_sheets.json
        <prefix>_gvd_sheets.obj
        <prefix>_gvd_sheets.glb
    """
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    paths = {}

    source_path = out_prefix.with_name(out_prefix.name + "_source_planes.json")
    source_payload = {int(gid): asdict(plane) for gid, plane in sorted(planes.items())}
    source_path.write_text(json.dumps(source_payload, indent=2))
    paths["source_planes_json"] = str(source_path)

    sheets_path = out_prefix.with_name(out_prefix.name + "_gvd_sheets.json")
    sheets_payload = [asdict(s) for s in sheets]
    sheets_path.write_text(json.dumps(sheets_payload, indent=2))
    paths["gvd_sheets_json"] = str(sheets_path)

    obj_path = out_prefix.with_name(out_prefix.name + "_gvd_sheets.obj")
    export_sheets_obj(sheets, obj_path)
    paths["gvd_sheets_obj"] = str(obj_path)

    glb_path = out_prefix.with_name(out_prefix.name + "_gvd_sheets.glb")
    export_sheets_glb(sheets, glb_path)
    paths["gvd_sheets_glb"] = str(glb_path)

    return paths


def export_sheets_obj(sheets: List[PlanarGVDSheet], path: str | Path) -> None:
    """
    Exports planar GVD sheets as OBJ polygons.

    The OBJ contains actual polygon faces, not voxel quads.
    Each object name encodes the source face-pair and branch.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        f.write("# Analytic planar GVD sheets\n")
        f.write("# Each face is a clipped angle-bisector plane primitive.\n")

        vertex_offset = 1

        for si, sheet in enumerate(sheets):
            poly = np.asarray(sheet.polygon, dtype=np.float64)
            if len(poly) < 3:
                continue

            ga, gb = sheet.pair
            f.write(f"\no sheet_{si}_pair_{ga}_{gb}_{sheet.branch}\n")
            f.write(
                f"# plane: {sheet.normal[0]} {sheet.normal[1]} "
                f"{sheet.normal[2]} {sheet.offset}\n"
            )

            for p in poly:
                f.write(f"v {p[0]} {p[1]} {p[2]}\n")

            ids = list(range(vertex_offset, vertex_offset + len(poly)))
            f.write("f " + " ".join(map(str, ids)) + "\n")

            vertex_offset += len(poly)


def export_sheets_glb(sheets: List[PlanarGVDSheet], path: str | Path) -> None:
    """
    Exports a triangulated visualization mesh of the planar sheets.

    The underlying representation remains analytic in the JSON export.
    This GLB is only for viewing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    vertices = []
    faces = []
    face_colors = []

    for sheet in sheets:
        poly = np.asarray(sheet.polygon, dtype=np.float64)
        if len(poly) < 3:
            continue

        start = len(vertices)
        vertices.extend(poly.tolist())

        color = _color_for_pair_and_branch(sheet.pair, sheet.branch)

        # Fan triangulation of one planar polygon.
        for k in range(1, len(poly) - 1):
            faces.append([start, start + k, start + k + 1])
            face_colors.append(color)

    if not faces:
        mesh = trimesh.Trimesh(
            vertices=np.empty((0, 3)),
            faces=np.empty((0, 3), dtype=np.int64),
            process=False,
        )
    else:
        mesh = trimesh.Trimesh(
            vertices=np.asarray(vertices, dtype=np.float64),
            faces=np.asarray(faces, dtype=np.int64),
            process=False,
        )
        mesh.visual.face_colors = np.asarray(face_colors, dtype=np.uint8)

    mesh.export(path)


def clip_plane_to_aabb(
    normal: np.ndarray,
    offset: float,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    *,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Intersects an infinite plane with an axis-aligned bounding box.

    Returns ordered polygon vertices on:
        normal · x + offset = 0
    """
    n = _normalize(np.asarray(normal, dtype=np.float64))
    d = float(offset)

    corners = np.array(
        [
            [bbox_min[0], bbox_min[1], bbox_min[2]],
            [bbox_max[0], bbox_min[1], bbox_min[2]],
            [bbox_min[0], bbox_max[1], bbox_min[2]],
            [bbox_max[0], bbox_max[1], bbox_min[2]],
            [bbox_min[0], bbox_min[1], bbox_max[2]],
            [bbox_max[0], bbox_min[1], bbox_max[2]],
            [bbox_min[0], bbox_max[1], bbox_max[2]],
            [bbox_max[0], bbox_max[1], bbox_max[2]],
        ],
        dtype=np.float64,
    )

    edges = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    pts = []

    for ia, ib in edges:
        p0 = corners[ia]
        p1 = corners[ib]

        s0 = float(np.dot(n, p0) + d)
        s1 = float(np.dot(n, p1) + d)

        if abs(s0) <= eps:
            pts.append(p0)
        if abs(s1) <= eps:
            pts.append(p1)

        if s0 * s1 < -eps * eps:
            t = s0 / (s0 - s1)
            p = p0 + t * (p1 - p0)
            pts.append(p)

    pts = _dedupe_points(pts, eps=1e-7)

    if len(pts) < 3:
        return np.empty((0, 3), dtype=np.float64)

    return order_points_on_plane(np.asarray(pts, dtype=np.float64), n)


def make_edge_bisector_strip(
    p0: np.ndarray,
    p1: np.ndarray,
    plane_normal: np.ndarray,
    plane_offset: float,
    *,
    half_width: float,
) -> Optional[np.ndarray]:
    """
    Creates a rectangular strip lying in the bisector plane and centered
    on one B-Rep boundary segment.

    This is useful when you want local topological GVD sheets instead of
    full bbox-clipped bisector planes.
    """
    n = _normalize(np.asarray(plane_normal, dtype=np.float64))
    d = float(plane_offset)

    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)

    # Project the source edge onto the bisector plane.
    p0 = p0 - (np.dot(n, p0) + d) * n
    p1 = p1 - (np.dot(n, p1) + d) * n

    edge = p1 - p0
    edge_len = np.linalg.norm(edge)
    if edge_len <= 1e-12:
        return None

    e = edge / edge_len

    # Direction lying in the bisector plane and perpendicular to the edge.
    w = np.cross(n, e)
    if np.linalg.norm(w) <= 1e-12:
        # Fallback: choose any stable direction in the plane.
        u, _ = plane_basis(n)
        w = u
    else:
        w = _normalize(w)

    hw = float(half_width)

    return np.asarray(
        [
            p0 - hw * w,
            p1 - hw * w,
            p1 + hw * w,
            p0 + hw * w,
        ],
        dtype=np.float64,
    )


def _bisector_planes(
    pi: PlanePrimitive,
    pj: PlanePrimitive,
    *,
    branches: str,
) -> List[Tuple[str, np.ndarray, float]]:
    ni = np.asarray(pi.normal, dtype=np.float64)
    nj = np.asarray(pj.normal, dtype=np.float64)
    di = float(pi.offset)
    dj = float(pj.offset)

    candidates = []

    requested = []
    if branches in {"both", "minus"}:
        requested.append("minus")
    if branches in {"both", "plus"}:
        requested.append("plus")

    for branch in requested:
        if branch == "minus":
            # phi_i - phi_j = 0
            n = ni - nj
            d = di - dj
        else:
            # phi_i + phi_j = 0
            n = ni + nj
            d = di + dj

        scale = np.linalg.norm(n)

        # Degenerate case: nearly identical oriented planes for the minus
        # branch, or opposite oriented planes for the plus branch.
        if scale <= 1e-12:
            continue

        n = n / scale
        d = d / scale

        candidates.append((branch, n, float(d)))

    return candidates


def order_points_on_plane(points: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """
    Orders coplanar polygon vertices counterclockwise in a local 2D basis.
    """
    n = _normalize(normal)
    c = points.mean(axis=0)

    u, v = plane_basis(n)
    rel = points - c[None, :]

    x = rel @ u
    y = rel @ v
    angles = np.arctan2(y, x)

    order = np.argsort(angles)
    return points[order]


def plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns orthonormal basis vectors u, v spanning the plane with given normal.
    """
    n = _normalize(normal)

    if abs(n[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])

    u = np.cross(n, ref)
    u = _normalize(u)
    v = np.cross(n, u)
    v = _normalize(v)

    return u, v


def _dedupe_points(points: Sequence[np.ndarray], eps: float) -> List[np.ndarray]:
    unique = []

    for p in points:
        p = np.asarray(p, dtype=np.float64)
        duplicate = False

        for q in unique:
            if np.linalg.norm(p - q) <= eps:
                duplicate = True
                break

        if not duplicate:
            unique.append(p)

    return unique


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n <= 1e-20:
        raise ValueError("Cannot normalize near-zero vector.")
    return v / n


def _canonical_pair(a: int, b: int) -> Pair:
    a = int(a)
    b = int(b)
    return (a, b) if a <= b else (b, a)


def _color_for_pair_and_branch(pair: Pair, branch: str) -> np.ndarray:
    """
    Stable pseudo-random RGBA color.
    """
    seed = (
        pair[0] * 73856093
        ^ pair[1] * 19349663
        ^ (1 if branch == "plus" else 2) * 83492791
    ) & 0xFFFFFFFF

    rng = np.random.default_rng(seed)
    rgb = rng.integers(60, 256, size=3, dtype=np.uint8)
    return np.array([rgb[0], rgb[1], rgb[2], 185], dtype=np.uint8)


def export_segments_obj(segments: np.ndarray, out_path: str | Path) -> None:
    """
    Exports line segments as an OBJ containing `l` records.
    Useful for visualizing detected B-Rep boundary edges.
    """
    segments = np.asarray(segments, dtype=np.float64).reshape(-1, 2, 3)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        f.write("# B-Rep boundary segments\n")
        vid = 1
        for p0, p1 in segments:
            f.write(f"v {p0[0]} {p0[1]} {p0[2]}\n")
            f.write(f"v {p1[0]} {p1[1]} {p1[2]}\n")
            f.write(f"l {vid} {vid + 1}\n")
            vid += 2
