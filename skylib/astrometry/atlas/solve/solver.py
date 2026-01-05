"""Assisted plate solver using UCAC4 catalog zones."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from skylib.astrometry.atlas.catalog.ucac4 import Ucac4Index
from skylib.astrometry.atlas.extract.sources import ExtractedSources, extract_sources
from skylib.astrometry.atlas.match.triangles import TriangleSet, build_kdtree, sample_triangles
from skylib.astrometry.atlas.wcs.build import wcs_from_similarity

try:  # pragma: no cover - optional dependency
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    from skylib.astrometry.atlas.match.triangles import cKDTree


@dataclass
class SolveResult:
    success: bool
    wcs: Optional[WCS]
    metadata: dict


def solve_assisted(
    fits_path: Path,
    ucac4_root: Path,
    *,
    ra0_deg: float,
    dec0_deg: float,
    scale_range_arcsec_per_pix: Tuple[float, float],
    fov_guess_deg: Optional[Tuple[float, float]] = None,
    timeout_s: Optional[float] = None,
    max_catalog_stars: int = 400,
    max_image_stars: int = 120,
    n_tri_obs: int = 8000,
    n_tri_cat: int = 15000,
    invariant_tol: float = 0.006,
    match_tol_arcsec: float = 6.0,
    refine_center: bool = True,
    thin: int = 1,
) -> SolveResult:
    start = time.perf_counter()

    sources = extract_sources(
        fits_path,
        max_sources=max_image_stars,
        crop_fraction=0.8,
        downsample=1,
    )
    obs_xy = sources.xy
    height, width = sources.shape
    if obs_xy.size == 0:
        return SolveResult(False, None, {"reason": "no_sources"})

    if fov_guess_deg is None:
        fov_guess_deg = _estimate_fov(fits_path, width, height, scale_range_arcsec_per_pix)
    if fov_guess_deg is None:
        return SolveResult(False, None, {"reason": "missing_fov"})

    ra_width, dec_height = fov_guess_deg
    pad = 1.3
    ra_width *= pad
    dec_height *= pad

    cos_dec = np.cos(np.deg2rad(dec0_deg))
    cos_dec = max(0.2, float(abs(cos_dec)))
    ra_half = (ra_width / cos_dec) / 2.0
    dec_half = dec_height / 2.0

    ucac4 = Ucac4Index(ucac4_root)
    cat = ucac4.query_box(
        ra0_deg - ra_half,
        ra0_deg + ra_half,
        dec0_deg - dec_half,
        dec0_deg + dec_half,
        thin=thin,
    )
    if cat.ra_deg.size == 0:
        return SolveResult(False, None, {"reason": "empty_catalog"})

    cat_xy = _gnomonic_projection(cat.ra_deg, cat.dec_deg, ra0_deg, dec0_deg)
    cat_xy, cat_radec = _limit_catalog(cat_xy, cat.ra_deg, cat.dec_deg, max_catalog_stars)

    if len(cat_xy) < 3:
        return SolveResult(False, None, {"reason": "insufficient_catalog"})

    rng = np.random.default_rng(0)
    min_scale, max_scale = scale_range_arcsec_per_pix
    a_min = np.deg2rad(min_scale / 3600.0)
    a_max = np.deg2rad(max_scale / 3600.0)

    min_side_pix = max(5.0, 0.02 * min(width, height))
    max_side_pix = 0.6 * max(width, height)
    min_side_rad = min_side_pix * a_min
    max_side_rad = max_side_pix * a_max

    obs_tri = sample_triangles(
        obs_xy,
        n_tri_obs,
        min_side=min_side_pix,
        max_side=max_side_pix,
        rng=rng,
    )
    cat_tri = sample_triangles(
        cat_xy,
        n_tri_cat,
        min_side=min_side_rad,
        max_side=max_side_rad,
        rng=rng,
    )

    if len(obs_tri.triangles) == 0 or len(cat_tri.triangles) == 0:
        return SolveResult(False, None, {"reason": "no_triangles"})

    inv_tree = build_kdtree(cat_tri.invariants)
    cat_tree = cKDTree(cat_xy)
    tol_rad = np.deg2rad(match_tol_arcsec / 3600.0)

    best = _match_triangles(
        obs_tri,
        cat_tri,
        inv_tree,
        cat_tree,
        obs_xy,
        a_min,
        a_max,
        tol_rad,
        timeout_s=timeout_s,
        start=start,
        invariant_tol=invariant_tol,
    )
    if best is None:
        return SolveResult(False, None, {"reason": "no_match"})

    scale, rotation, translation, inliers, rms = best
    wcs = wcs_from_similarity(scale, rotation, translation, ra0_deg, dec0_deg)

    if refine_center:
        center_x = (width + 1) / 2.0
        center_y = (height + 1) / 2.0
        new_ra, new_dec = wcs.all_pix2world([[center_x, center_y]], 1)[0]
        wcs = _refine_center(obs_xy, cat_radec, inliers, scale, rotation, translation, new_ra, new_dec)
        ra0_deg, dec0_deg = new_ra, new_dec

    elapsed = time.perf_counter() - start
    metadata = {
        "ra0_deg": float(ra0_deg),
        "dec0_deg": float(dec0_deg),
        "scale_arcsec_per_pix": float(scale * (180.0 / np.pi) * 3600.0),
        "rotation_deg": float(np.rad2deg(np.arctan2(rotation[1, 0], rotation[0, 0]))),
        "rms_arcsec": float(rms * (180.0 / np.pi) * 3600.0),
        "inliers": int(inliers),
        "match_method": "triangles",
        "elapsed_s": float(elapsed),
    }
    return SolveResult(True, wcs, metadata)


def _estimate_fov(
    fits_path: Path,
    width: int,
    height: int,
    scale_range_arcsec_per_pix: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
    try:
        with fits.open(fits_path) as hdul:
            header = hdul[0].header
    except Exception:
        header = None

    if header is not None:
        try:
            wcs = WCS(header)
            scales = proj_plane_pixel_scales(wcs)
            if scales is not None and len(scales) >= 2:
                fov_w = abs(scales[0]) * width
                fov_h = abs(scales[1]) * height
                if fov_w > 0 and fov_h > 0:
                    return (float(fov_w), float(fov_h))
        except Exception:
            pass

    min_scale, max_scale = scale_range_arcsec_per_pix
    scale = 0.5 * (min_scale + max_scale)
    if scale <= 0:
        return None
    fov_w = scale * width / 3600.0
    fov_h = scale * height / 3600.0
    return (float(fov_w), float(fov_h))


def _gnomonic_projection(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    ra0_deg: float,
    dec0_deg: float,
) -> np.ndarray:
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    ra0 = np.deg2rad(ra0_deg)
    dec0 = np.deg2rad(dec0_deg)

    cosc = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(ra - ra0)
    xi = np.cos(dec) * np.sin(ra - ra0) / cosc
    eta = (np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)) / cosc
    return np.stack([xi, eta], axis=1)


def _limit_catalog(
    cat_xy: np.ndarray,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    max_catalog_stars: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(cat_xy) <= max_catalog_stars:
        return cat_xy, np.stack([ra_deg, dec_deg], axis=1)
    dist = np.hypot(cat_xy[:, 0], cat_xy[:, 1])
    order = np.argsort(dist)
    order = order[:max_catalog_stars]
    return cat_xy[order], np.stack([ra_deg[order], dec_deg[order]], axis=1)


def _match_triangles(
    obs_tri: TriangleSet,
    cat_tri: TriangleSet,
    inv_tree: cKDTree,
    cat_tree: cKDTree,
    obs_xy: np.ndarray,
    a_min: float,
    a_max: float,
    tol_rad: float,
    *,
    timeout_s: Optional[float],
    start: float,
    invariant_tol: float,
):
    best = None
    best_inliers = 0
    best_rms = np.inf

    for obs_inv, obs_order in zip(obs_tri.invariants, obs_tri.ordered_points):
        if timeout_s is not None and (time.perf_counter() - start) > timeout_s:
            break
        candidate_idx = inv_tree.query_ball_point(obs_inv, r=invariant_tol)
        if not candidate_idx:
            continue
        for idx in candidate_idx:
            cat_order = cat_tri.ordered_points[idx]
            scale, rotation, translation = _fit_similarity(obs_order, cat_order)
            if scale < a_min or scale > a_max:
                continue
            inliers, rms = _score_candidate(obs_xy, cat_tree, scale, rotation, translation, tol_rad)
            if inliers > best_inliers or (inliers == best_inliers and rms < best_rms):
                best_inliers = inliers
                best_rms = rms
                best = (scale, rotation, translation, inliers, rms)
            if best_inliers >= max(12, int(0.4 * len(obs_xy))):
                return best
    return best


def _score_candidate(
    obs_xy: np.ndarray,
    cat_tree: cKDTree,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
    tol_rad: float,
) -> Tuple[int, float]:
    pred = (obs_xy @ rotation.T) * scale + translation
    dist, idx = cat_tree.query(pred)
    mask = dist <= tol_rad
    if not np.any(mask):
        return 0, float("inf")
    rms = float(np.sqrt(np.mean(dist[mask] ** 2)))
    return int(np.sum(mask)), rms


def _fit_similarity(src: np.ndarray, dst: np.ndarray):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean

    cov = dst_c.T @ src_c
    u, s, vt = np.linalg.svd(cov)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt

    var = np.sum(src_c ** 2)
    scale = np.sum(s) / var
    translation = dst_mean - scale * (rotation @ src_mean)
    return scale, rotation, translation


def _refine_center(
    obs_xy: np.ndarray,
    cat_radec: np.ndarray,
    inliers: int,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
    ra0_deg: float,
    dec0_deg: float,
) -> WCS:
    pred = (obs_xy @ rotation.T) * scale + translation
    cat_xy = _gnomonic_projection(cat_radec[:, 0], cat_radec[:, 1], ra0_deg, dec0_deg)
    cat_tree = cKDTree(cat_xy)
    dist, idx = cat_tree.query(pred)
    mask = dist <= np.median(dist) * 1.5
    if np.sum(mask) < max(6, int(0.5 * inliers)):
        return wcs_from_similarity(scale, rotation, translation, ra0_deg, dec0_deg)

    matched_obs = obs_xy[mask]
    matched_cat = cat_xy[idx[mask]]
    scale, rotation, translation = _fit_similarity(matched_obs, matched_cat)
    return wcs_from_similarity(scale, rotation, translation, ra0_deg, dec0_deg)
