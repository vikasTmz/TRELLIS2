import numpy as np
import trimesh
from skimage import measure
from tqdm.auto import tqdm

import time
from functools import wraps
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


def sample_and_export_boundary_points(
    boundary_segments,
    output_path=None,
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

    # Export using Trimesh PointCloud
    if output_path is not None:
        pc = trimesh.points.PointCloud(final_points)
        pc.visual.vertex_colors = [255, 0, 0, 255]
        pc.export(output_path)
        print(f"Successfully exported {len(final_points)} points to {output_path}")

    return final_points


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

    nx, ny, nz = len(xs), len(ys), len(zs)

    m = np.full((nx, ny, nz), np.float32(1e6), dtype=np.float32)
    s = np.zeros((nx, ny, nz), dtype=np.float32)

    xs_f = xs.astype(np.float32, copy=False)
    ys_f = ys.astype(np.float32, copy=False)
    zs_f = zs.astype(np.float32, copy=False)

    radius = np.float32(radius)
    k = np.float32(k)

    if weights is None:
        weights = np.ones(len(points), dtype=np.float32)
    else:
        weights = np.asarray(weights, dtype=np.float32)

    pts = points.astype(np.float32, copy=False)

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

        scale = np.exp(-k * (m0 - new_m)).astype(np.float32, copy=False)
        contrib = (w * np.exp(-k * (di - new_m))).astype(np.float32, copy=False)

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


def export_mesh(mesh: trimesh.Trimesh, out_ply: str, out_glb: str) -> None:
    if out_ply is not None:
        mesh.export(out_ply)  # PLY
    # GLB export: wrap in a scene for best compatibility
    scene = trimesh.Scene(mesh)
    scene.export(out_glb)
