"""Assisted plate solver using UCAC catalog zones."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from skylib.astrometry.atlas.catalog import CatalogIndex, get_catalog_spec
from skylib.astrometry.atlas.config import AtlasConfig
from skylib.astrometry.atlas.extract.sources import ExtractedSources, extract_sources
from skylib.astrometry.atlas.match.triangles import TriangleSet, build_kdtree, sample_triangles
from skylib.astrometry.atlas.wcs.build import wcs_from_similarity

try:  # pragma: no cover - optional dependency
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    from skylib.astrometry.atlas.match.triangles import cKDTree

ARCSEC_TO_RAD = np.deg2rad(1.0 / 3600.0)

@dataclass
class SolveResult:
    success: bool
    wcs: Optional[WCS]
    metadata: dict

def _verify_candidate(
    obs_xy: np.ndarray,
    cat_tree: cKDTree,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
    *,
    tol_arcsec: float,
) -> tuple[float, int, float]:
    """Re-score a candidate similarity transform at a given tolerance.

    Returns: (score, inliers, rms_arcsec)
    """
    tol_rad = tol_arcsec * ARCSEC_TO_RAD
    score, inliers, rms_rad = _score_candidate(
        obs_xy, cat_tree, scale, rotation, translation, tol_rad
    )
    rms_arcsec = float(rms_rad / ARCSEC_TO_RAD)
    return float(score), int(inliers), rms_arcsec

def solve(
    fits_path: Path,
    config: AtlasConfig,
    *,
    ra0_deg: float,
    dec0_deg: float,
    scale_range_arcsec_per_pix: Tuple[float, float],
    fov_guess_deg: Optional[Tuple[float, float]] = None,
) -> SolveResult:
    start = time.perf_counter()

    debug_overlay_path: Optional[Path] = None
    if config.debug:
        debug_overlay_path = fits_path.with_suffix(".png").with_name(fits_path.stem + "_sources.png")

    sources = extract_sources(
        fits_path,
        max_sources=config.max_image_stars,
        crop_fraction=1.0,
        downsample=1,
        edge_margin=8,
        sn_thresh=5.0,
        peak_sn_thresh=8.0,
        min_area=5,
        max_elong=50.0,   # if you want to tolerate long trails
        debug_overlay_path=debug_overlay_path,
    )
    obs_xy = sources.xy
    height, width = sources.shape

    # ---- Dense-field handling: match/verify only on brightest sources ----
    # In globular clusters / crowded fields, many detections are real but *not* in the catalog
    # (too faint, blended, missing from UCAC5). Using all sources makes "inlier fraction"
    # meaningless and increases false local correspondences.
    max_match_sources = int(getattr(config, "max_match_sources", 180))
    if obs_xy.shape[0] > max_match_sources and hasattr(sources, "flux") and sources.flux.size == obs_xy.shape[0]:
        order = np.argsort(sources.flux)[::-1]  # brightest first
        keep = order[:max_match_sources]
        obs_xy_match = obs_xy[keep]
    else:
        obs_xy_match = obs_xy

    if config.debug:
        print(f"obs sources (raw): n={len(obs_xy)} ; using for match/verify: n={len(obs_xy_match)}")


    if obs_xy_match.size == 0:
        return SolveResult(False, None, {"reason": "no_sources"})

    if fov_guess_deg is None:
        fov_guess_deg = _estimate_fov(fits_path, width, height, scale_range_arcsec_per_pix)
    if fov_guess_deg is None:
        return SolveResult(False, None, {"reason": "missing_fov"})
    
    print(f"FOV GUESS: {fov_guess_deg}")

    # ---- Stage 0: single catalog query with conservative padded footprint ----
    catalog_pad_frac = config.catalog_pad_frac
    catalog_max_radius_deg = config.catalog_max_radius_deg

    half_diag_deg = _search_half_diag_deg(
        width,
        height,
        scale_range_arcsec_per_pix,
        fov_guess_deg,
        pad_frac=catalog_pad_frac,
    )
    if half_diag_deg <= 0:
        return SolveResult(False, None, {"reason": "bad_fov_or_scale"})

    if catalog_max_radius_deg is not None:
        half_diag_deg = min(float(half_diag_deg), float(catalog_max_radius_deg))

    # Convert "radius" into a RA/Dec box (box is conservative; good for your zone-based catalog query)
    cos_dec = np.cos(np.deg2rad(dec0_deg))
    cos_dec = max(0.2, float(abs(cos_dec)))

    dec_half = half_diag_deg
    ra_half = half_diag_deg / cos_dec

    print(f"searching: {(ra0_deg - ra_half, ra0_deg + ra_half, dec0_deg - dec_half, dec0_deg + dec_half)}")

    catalog_name, catalog_root = config.resolve_catalog()
    catalog_index = _catalog_index(catalog_name, catalog_root)
    cat = catalog_index.query_box(
        ra0_deg - ra_half,
        ra0_deg + ra_half,
        dec0_deg - dec_half,
        dec0_deg + dec_half,
        thin=config.thin,
    )
    if cat.ra_deg.size == 0:
        return SolveResult(False, None, {"reason": "empty_catalog"})

    cat_xy = _gnomonic_projection(cat.ra_deg, cat.dec_deg, ra0_deg, dec0_deg)
    cat_xy, cat_radec = _limit_catalog(
        cat_xy,
        cat.ra_deg,
        cat.dec_deg,
        config.max_catalog_stars,
    )

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
    min_side_pix = 30
    max_side_pix = 2000

    obs_tri = sample_triangles(
        obs_xy_match,
        config.n_tri_obs,
        min_side=min_side_pix,
        max_side=max_side_pix,
        rng=rng,
    )
    cat_tri = sample_triangles(
        cat_xy,
        config.n_tri_cat,
        min_side=min_side_rad,
        max_side=max_side_rad,
        rng=rng,
    )

    if len(obs_tri.triangles) == 0 or len(cat_tri.triangles) == 0:
        return SolveResult(False, None, {"reason": "no_triangles"})

    inv_tree = build_kdtree(cat_tri.invariants)
    cat_tree = cKDTree(cat_xy)
    tol_rad = np.deg2rad(config.match_tol_arcsec / 3600.0)

    print(f"calling match triangles: (obs_tri={len(obs_tri.triangles)}, cat_tri={len(cat_tri.triangles)})")
    print(f"obs sources: n={len(obs_xy_match)}  tol_arcsec={config.match_tol_arcsec}  invariant_tol={config.invariant_tol}")
    print(f"scale gate: a_min={a_min:.3e} rad/pix  a_max={a_max:.3e} rad/pix  (min={min_scale} max={max_scale} arcsec/pix)")
    print(f"cat stars used: n={len(cat_xy)}  (max_catalog_stars={config.max_catalog_stars})")

    best = _match_triangles(
        obs_tri,
        cat_tri,
        inv_tree,
        cat_tree,
        obs_xy_match,
        a_min,
        a_max,
        tol_rad,
        timeout_s=config.timeout_s,
        start=start,
        invariant_tol=config.invariant_tol,
        debug=config.debug
    )
    if best is None:
        return SolveResult(False, None, {"reason": "no_match"})

    scale, rotation, translation, _, _ = best

    # --------------------------
    # COARSE verification (your existing tolerance)
    # --------------------------
    coarse_score, coarse_inliers, coarse_rms_arcsec = _verify_candidate(
        obs_xy_match,
        cat_tree,
        scale,
        rotation,
        translation,
        tol_arcsec=float(config.match_tol_arcsec),
    )
    coarse_frac = coarse_inliers / max(len(obs_xy_match), 1)

    # Coarse gates (still permissive)
    COARSE_MIN_INLIERS = 8
    COARSE_MIN_FRAC = 0.15
    COARSE_MAX_RMS_ARCSEC = 2.5

    if config.debug:
        print(
            "verify coarse:",
            {
                "score": coarse_score,
                "inliers": coarse_inliers,
                "n_obs": int(len(obs_xy_match)),
                "frac": float(coarse_frac),
                "rms_arcsec": float(coarse_rms_arcsec),
                "tol_arcsec": float(config.match_tol_arcsec),
            },
        )

    if (
        coarse_inliers < COARSE_MIN_INLIERS
        or coarse_frac < COARSE_MIN_FRAC
        or coarse_rms_arcsec > COARSE_MAX_RMS_ARCSEC
        or not np.isfinite(coarse_score)
    ):
        if config.debug:
            print(
                f"Rejecting candidate at coarse verify: "
                f"inliers={coarse_inliers} rms={coarse_rms_arcsec:.3f} arcsec"
            )
        return SolveResult(False, None, {"reason": "no_confident_match_coarse"})

    # --------------------------
    # TIGHT verification (strong check)
    # --------------------------
    TIGHT_TOL_ARCSEC = 1.0
    tight_score, tight_inliers, tight_rms_arcsec = _verify_candidate(
        obs_xy_match,
        cat_tree,
        scale,
        rotation,
        translation,
        tol_arcsec=TIGHT_TOL_ARCSEC,
    )
    tight_frac = tight_inliers / max(len(obs_xy_match), 1)

    TIGHT_MIN_INLIERS = 10
    TIGHT_MIN_FRAC = 0.15
    TIGHT_MIN_INLIERS_ABS_OK = 18
    
    TIGHT_MAX_RMS_ARCSEC = 1.5

    if config.debug:
        print(
            "verify tight:",
            {
                "score": tight_score,
                "inliers": tight_inliers,
                "n_obs": int(len(obs_xy_match)),
                "frac": float(tight_frac),
                "rms_arcsec": float(tight_rms_arcsec),
                "tol_arcsec": float(TIGHT_TOL_ARCSEC),
            },
        )

    if (
        tight_inliers < TIGHT_MIN_INLIERS
        or (tight_frac < TIGHT_MIN_FRAC and tight_inliers < TIGHT_MIN_INLIERS_ABS_OK)
        or tight_rms_arcsec > TIGHT_MAX_RMS_ARCSEC
        or not np.isfinite(tight_score)
    ):
        # Optional: a mid-tier fallback can help if centroiding is ~1"
        MID_TOL_ARCSEC = 1.5
        mid_score, mid_inliers, mid_rms_arcsec = _verify_candidate(
            obs_xy_match,
            cat_tree,
            scale,
            rotation,
            translation,
            tol_arcsec=MID_TOL_ARCSEC,
        )
        mid_frac = mid_inliers / max(len(obs_xy_match), 1)

        if config.debug:
            print(
                "verify mid:",
                {
                    "score": mid_score,
                    "inliers": mid_inliers,
                    "n_obs": int(len(obs_xy_match)),
                    "frac": float(mid_frac),
                    "rms_arcsec": float(mid_rms_arcsec),
                    "tol_arcsec": float(MID_TOL_ARCSEC),
                },
            )

        MID_MIN_INLIERS = 8
        MID_MIN_FRAC = 0.20
        MID_MAX_RMS_ARCSEC = 1.7

        if (
            mid_inliers < MID_MIN_INLIERS
            or mid_frac < MID_MIN_FRAC
            or mid_rms_arcsec > MID_MAX_RMS_ARCSEC
            or not np.isfinite(mid_score)
        ):
            if config.debug:
                print(
                    "Rejecting candidate at tight verify: "
                    f"inliers={tight_inliers} rms={tight_rms_arcsec:.3f} arcsec"
                )
            return SolveResult(
                False,
                None,
                {
                    "reason": "verification_failed",
                    "coarse": {
                        "score": coarse_score,
                        "inliers": coarse_inliers,
                        "frac": float(coarse_frac),
                        "rms_arcsec": coarse_rms_arcsec,
                        "tol_arcsec": float(config.match_tol_arcsec),
                    },
                    "tight": {
                        "score": tight_score,
                        "inliers": tight_inliers,
                        "frac": float(tight_frac),
                        "rms_arcsec": tight_rms_arcsec,
                        "tol_arcsec": float(TIGHT_TOL_ARCSEC),
                    },
                    "mid": {
                        "score": mid_score,
                        "inliers": mid_inliers,
                        "frac": float(mid_frac),
                        "rms_arcsec": mid_rms_arcsec,
                        "tol_arcsec": float(MID_TOL_ARCSEC),
                    },
                },
            )

    # If we get here, accept the candidate.
    # For metadata, use the tight or mid values (prefer tight if it passed).
    inliers = tight_inliers
    rms_arcsec = tight_rms_arcsec
    
    wcs = wcs_from_similarity(scale, rotation, translation, ra0_deg, dec0_deg)

    # NOTE: DO NOT rebuild the WCS with a different (ra0,dec0) unless you also
    # reproject the catalog to that new tangent point and re-fit translation.
    if config.refine_center:
        # Astropy pixel_to_world_values expects 0-based pixel coords
        cx0 = (width - 1) / 2.0
        cy0 = (height - 1) / 2.0
        center_ra, center_dec = wcs.pixel_to_world_values(cx0, cy0)

        if config.debug:
            print(f"Refined center (from WCS @ image center): RA={center_ra:.6f} DEC={center_dec:.6f}")

        # Store refined center in metadata only
        refined_center_ra_deg = float(center_ra) % 360.0
        refined_center_dec_deg = float(center_dec)
    else:
        refined_center_ra_deg = None
        refined_center_dec_deg = None

    elapsed = time.perf_counter() - start
    metadata = {
        "ra0_deg": float(ra0_deg),
        "dec0_deg": float(dec0_deg),
        "refined_center_ra_deg": refined_center_ra_deg,
        "refined_center_dec_deg": refined_center_dec_deg,
        "scale_arcsec_per_pix": float(scale * (180.0 / np.pi) * 3600.0),
        "rotation_deg": float(np.rad2deg(np.arctan2(rotation[1, 0], rotation[0, 0]))),
        "rms_arcsec": float(rms_arcsec),
        "inliers": int(inliers),
        "catalog": catalog_name,
        "match_method": "triangles",
        "elapsed_s": float(elapsed),
    }
    
    return SolveResult(True, wcs, metadata)

def _search_half_diag_deg(
    width: int,
    height: int,
    scale_range_arcsec_per_pix: Tuple[float, float],
    fov_guess_deg: Optional[Tuple[float, float]],
    *,
    pad_frac: float,
) -> float:
    """
    Compute a conservative half-diagonal FOV radius (deg) to use for Stage 0 catalog search.

    - Uses max scale to avoid underestimating sky footprint.
    - Falls back to fov_guess if provided.
    - Applies multiplicative padding (1 + pad_frac).
    """
    min_scale, max_scale = scale_range_arcsec_per_pix

    # Prefer geometry from scale bounds (more reliable than fov_guess)
    if max_scale > 0:
        fov_w = (max_scale * width) / 3600.0  # deg
        fov_h = (max_scale * height) / 3600.0
    elif fov_guess_deg is not None:
        fov_w, fov_h = fov_guess_deg
    else:
        return 0.0

    half_diag = 0.5 * float(np.hypot(fov_w, fov_h))
    return half_diag * (1.0 + float(pad_frac))

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
    debug: bool = False,
):
    """Search triangle-invariant matches and return the best similarity transform.

    Debug counters explain where candidates are being lost:
      - invariant hits (query_ball_point empty vs non-empty)
      - scale gate rejections
      - scoring producing -inf (no inliers)
      - number of best updates
    """
    best = None
    best_score = float("-inf")
    best_inliers = 0
    best_rms = np.inf

    # ---- debug counters ----
    n_obs_inv = 0
    n_obs_inv_nonempty = 0
    n_candidate_pairs = 0
    n_scale_reject = 0
    n_scale_ok = 0
    n_scored = 0
    n_scored_finite = 0
    n_scored_noninf = 0
    best_updates = 0
    timed_out = False

    # Optional: track top few candidates by score for inspection
    topk = []  # list of (score, inliers, rms, scale)

    for obs_inv, obs_order in zip(obs_tri.invariants, obs_tri.ordered_points):
        n_obs_inv += 1

        if timeout_s is not None and (time.perf_counter() - start) > timeout_s:
            timed_out = True
            break

        candidate_idx = inv_tree.query_ball_point(obs_inv, r=invariant_tol)
        if not candidate_idx:
            continue

        n_obs_inv_nonempty += 1
        n_candidate_pairs += len(candidate_idx)

        for idx in candidate_idx:
            cat_order = cat_tri.ordered_points[idx]
            scale, rotation, translation = _fit_similarity(obs_order, cat_order)

            if scale < a_min or scale > a_max:
                n_scale_reject += 1
                continue

            n_scale_ok += 1

            score, inliers, rms = _score_candidate(
                obs_xy, cat_tree, scale, rotation, translation, tol_rad
            )
            n_scored += 1

            if np.isfinite(score):
                n_scored_finite += 1
            if score != float("-inf"):
                n_scored_noninf += 1

            # keep a small top-k list
            if np.isfinite(score):
                topk.append((float(score), int(inliers), float(rms), float(scale)))
                if len(topk) > 50:
                    # keep only top 10 by score to avoid unbounded growth
                    topk.sort(key=lambda t: t[0], reverse=True)
                    topk = topk[:10]

            if (
                score > best_score
                or (score == best_score and inliers > best_inliers)
                or (score == best_score and inliers == best_inliers and rms < best_rms)
            ):
                best_score = float(score)
                best_inliers = int(inliers)
                best_rms = float(rms)
                best = (scale, rotation, translation, inliers, rms)
                best_updates += 1
                if debug:
                    # Print enough to identify if this is "real-ish"
                    print(
                        "new best:",
                        {
                            "score": best_score,
                            "inliers": best_inliers,
                            "rms_arcsec": best_rms * (180.0 / np.pi) * 3600.0,
                            "scale_arcsec_per_pix": float(scale * (180.0 / np.pi) * 3600.0),
                        },
                    )

    if debug:
        print(
            "match_triangles summary:",
            {
                "obs_triangles": int(len(obs_tri.triangles)),
                "cat_triangles": int(len(cat_tri.triangles)),
                "obs_invariants_total": int(n_obs_inv),
                "obs_invariants_with_hits": int(n_obs_inv_nonempty),
                "cand_pairs_total": int(n_candidate_pairs),
                "scale_ok": int(n_scale_ok),
                "scale_reject": int(n_scale_reject),
                "scored": int(n_scored),
                "scored_finite": int(n_scored_finite),
                "scored_noninf": int(n_scored_noninf),
                "best_updates": int(best_updates),
                "timed_out": bool(timed_out),
                "best": None
                if best is None
                else {
                    "best_score": float(best_score),
                    "best_inliers": int(best_inliers),
                    "best_rms_arcsec": float(best_rms * (180.0 / np.pi) * 3600.0),
                    "best_scale_arcsec_per_pix": float(best[0] * (180.0 / np.pi) * 3600.0),
                },
            },
        )
        if topk:
            topk.sort(key=lambda t: t[0], reverse=True)
            print("top candidates (score, inliers, rms_arcsec, scale_arcsec_per_pix):")
            for s, inl, rms, sc in topk[:10]:
                print(
                    "  ",
                    (
                        s,
                        inl,
                        rms * (180.0 / np.pi) * 3600.0,
                        sc * (180.0 / np.pi) * 3600.0,
                    ),
                )

    return best


def _score_candidate(
    obs_xy: np.ndarray,
    cat_tree: cKDTree,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
    tol_rad: float,
) -> Tuple[float, int, float]:
    """Score a candidate similarity transform.

    B) Enforces a one-to-one assignment between observed sources and catalog sources
       (greedy by smallest residual) to avoid inflated inlier counts from collisions.

    C) Uses a score that penalizes outliers and collisions, not just RMS-on-inliers.
       This makes false positives much less likely in dense fields.

    Returns: (score, inliers, rms_rad)
    """
    pred = (obs_xy @ rotation.T) * scale + translation
    dist, idx = cat_tree.query(pred)

    within = dist <= tol_rad
    if not np.any(within):
        return float("-inf"), 0, float("inf")

    d = dist[within]
    j = idx[within]

    # Greedy one-to-one assignment: keep the closest observed point for each catalog id.
    order = np.argsort(d)
    keep_mask = np.zeros_like(d, dtype=bool)
    seen = set()
    for k in order:
        cj = int(j[k])
        if cj in seen:
            continue
        seen.add(cj)
        keep_mask[k] = True

    inlier_dist = d[keep_mask]
    inliers = int(inlier_dist.size)
    if inliers == 0:
        return float("-inf"), 0, float("inf")

    # Collisions = extra obs that were within tol but mapped to an already-used catalog star.
    collisions = int(d.size - inlier_dist.size)
    outliers = int(obs_xy.shape[0] - inliers)

    rms = float(np.sqrt(np.mean(inlier_dist ** 2)))
    rms_norm = rms / max(float(tol_rad), 1e-12)

    # Higher is better. Strongly prefers: many inliers, low RMS, few collisions, few outliers.
    score = (
        inliers
        - 2.0 * collisions
        - 0.25 * outliers
        - 0.5 * (rms_norm ** 2) * inliers
    )

    return float(score), inliers, rms


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


def _catalog_index(catalog: str, root: Path) -> CatalogIndex:
    spec = get_catalog_spec(catalog)
    return spec.index_factory(root)
