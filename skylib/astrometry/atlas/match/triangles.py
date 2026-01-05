"""Triangle matching utilities for assisted plate solving."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - fallback implementation

    class cKDTree:  # type: ignore[no-redef]
        def __init__(self, data: np.ndarray):
            self.data = np.asarray(data, dtype=np.float64)

        def query(self, points: np.ndarray, k: int = 1):
            points = np.atleast_2d(points)
            diff = self.data[None, :, :] - points[:, None, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=2))
            idx = np.argmin(dist, axis=1)
            return dist[np.arange(len(points)), idx], idx

        def query_ball_point(self, point: np.ndarray, r: float):
            diff = self.data - point
            dist = np.sqrt(np.sum(diff ** 2, axis=1))
            return np.where(dist <= r)[0].tolist()


@dataclass
class TriangleSet:
    triangles: np.ndarray
    invariants: np.ndarray
    ordered_points: np.ndarray


def sample_triangles(
    points: np.ndarray,
    max_triangles: int,
    *,
    min_side: float,
    max_side: float,
    rng: np.random.Generator,
) -> TriangleSet:
    n_points = points.shape[0]
    if n_points < 3 or max_triangles <= 0:
        empty = np.empty((0, 3), dtype=np.int32)
        return TriangleSet(empty, np.empty((0, 2)), np.empty((0, 3, 2)))

    triangles: list[np.ndarray] = []
    invariants: list[np.ndarray] = []
    ordered_pts: list[np.ndarray] = []

    attempts = max_triangles * 20
    while len(triangles) < max_triangles and attempts > 0:
        attempts -= 1
        idx = rng.choice(n_points, size=3, replace=False)
        pts = points[idx]
        inv, ordered = triangle_invariant_and_order(pts, min_side=min_side, max_side=max_side)
        if inv is None:
            continue
        triangles.append(idx)
        invariants.append(inv)
        ordered_pts.append(ordered)

    if not triangles:
        empty = np.empty((0, 3), dtype=np.int32)
        return TriangleSet(empty, np.empty((0, 2)), np.empty((0, 3, 2)))

    return TriangleSet(
        triangles=np.asarray(triangles, dtype=np.int32),
        invariants=np.asarray(invariants, dtype=np.float64),
        ordered_points=np.asarray(ordered_pts, dtype=np.float64),
    )


def triangle_invariant_and_order(
    points: np.ndarray,
    *,
    min_side: float,
    max_side: float,
    collinear_eps: float = 1e-6,
) -> Tuple[np.ndarray | None, np.ndarray | None]:
    p0, p1, p2 = points
    d01 = np.linalg.norm(p0 - p1)
    d12 = np.linalg.norm(p1 - p2)
    d20 = np.linalg.norm(p2 - p0)

    if min(d01, d12, d20) < min_side or max(d01, d12, d20) > max_side:
        return None, None

    area = np.cross(p1 - p0, p2 - p0)
    if abs(area) < collinear_eps:
        return None, None

    sides = np.array([d01, d12, d20], dtype=np.float64)
    order = np.argsort(sides)
    l1, l2, l3 = sides[order]
    inv = np.array([l1 / l3, l2 / l3], dtype=np.float64)

    ordered = _canonical_order(points)
    return inv, ordered


def _canonical_order(points: np.ndarray) -> np.ndarray:
    p0, p1, p2 = points
    d01 = np.linalg.norm(p0 - p1)
    d12 = np.linalg.norm(p1 - p2)
    d20 = np.linalg.norm(p2 - p0)

    if d01 >= d12 and d01 >= d20:
        idx = (0, 1, 2)
    elif d12 >= d01 and d12 >= d20:
        idx = (1, 2, 0)
    else:
        idx = (2, 0, 1)

    ordered = points[list(idx)]
    cross = np.cross(ordered[1] - ordered[0], ordered[2] - ordered[0])
    if cross < 0:
        ordered = np.array([ordered[1], ordered[0], ordered[2]])
    return ordered


def build_kdtree(invariants: np.ndarray) -> cKDTree:
    if len(invariants) == 0:
        return cKDTree(np.empty((0, 2)))
    return cKDTree(invariants)
