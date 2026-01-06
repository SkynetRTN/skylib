from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover
    ndi = None


@dataclass
class ExtractedSources:
    xy: np.ndarray
    shape: Tuple[int, int]
    flux: np.ndarray          # brightness proxy (sum of background-subtracted positive flux)
    peak_sn: np.ndarray       # peak S/N in detection blob


def extract_sources(
    fits_path: Path,
    *,
    max_sources: int = 500,
    crop_fraction: float = 1.0,
    downsample: int = 1,
    edge_margin: int = 8,
    sn_thresh: float = 5.0,          # detection threshold in S/N
    peak_sn_thresh: float = 8.0,     # require at least this peak S/N in a blob
    min_area: int = 5,               # min connected pixels
    max_area: int = 10_000,          # reject huge blobs (saturation blooms, etc)
    max_elong: float = 20.0,         # allow trails; lower if you want
    debug_overlay_path: Optional[Path] = None,
) -> ExtractedSources:
    if ndi is None:
        raise ImportError("scipy is required for extract_sources_threshold")

    with fits.open(fits_path) as hdul:
        data = hdul[0].data

    if data is None:
        raise ValueError("FITS image contains no data")
    if data.ndim > 2:
        data = data[0]

    data = np.asarray(data, dtype=np.float64)
    full_shape = data.shape

    crop_fraction = max(0.1, min(1.0, float(crop_fraction)))
    downsample = max(1, int(downsample))

    y0, x0 = 0, 0
    if crop_fraction < 1.0:
        h, w = data.shape
        ch = int(h * crop_fraction)
        cw = int(w * crop_fraction)
        y0 = (h - ch) // 2
        x0 = (w - cw) // 2
        data = data[y0 : y0 + ch, x0 : x0 + cw]

    if downsample > 1:
        data = data[::downsample, ::downsample]

    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    std = max(float(std), 1e-12)

    sn = (data - float(median)) / std

    # Binary mask of candidate pixels
    mask = sn > float(sn_thresh)

    # Morphology: remove isolated pixels and smooth small holes
    # (3x3 structure is conservative)
    structure = np.ones((3, 3), dtype=bool)
    mask = ndi.binary_opening(mask, structure=structure)
    mask = ndi.binary_closing(mask, structure=structure)

    labeled, nlab = ndi.label(mask, structure=structure)
    if nlab == 0:
        return ExtractedSources(
            xy=np.empty((0, 2)),
            shape=full_shape,
            flux=np.empty((0,), dtype=np.float64),
            peak_sn=np.empty((0,), dtype=np.float64),
        )
    

    # Measure each component
    slices = ndi.find_objects(labeled)
    detections = []
    for lab, slc in enumerate(slices, start=1):
        if slc is None:
            continue
        region = (labeled[slc] == lab)
        npix = int(region.sum())
        if npix < int(min_area) or npix > int(max_area):
            continue

        sn_reg = sn[slc][region]
        peak_sn = float(sn_reg.max())
        if peak_sn < float(peak_sn_thresh):
            continue

        # intensity-weighted centroid on background-subtracted flux
        flux_reg = (data[slc] - float(median))[region]
        flux_reg = np.clip(flux_reg, 0.0, None)
        flux_sum = float(flux_reg.sum())
        if flux_sum <= 0:
            continue

        # Pixel coordinates within slice
        yy, xx = np.nonzero(region)
        # Convert to full-image coords in the downsampled/cropped space
        yy = yy + slc[0].start
        xx = xx + slc[1].start

        # Weighted centroid
        w = flux_reg
        cy = float((yy * w).sum() / flux_sum)
        cx = float((xx * w).sum() / flux_sum)

        # 2nd moments for elongation estimate
        dy = yy - cy
        dx = xx - cx
        mxx = float((w * dx * dx).sum() / flux_sum)
        myy = float((w * dy * dy).sum() / flux_sum)
        mxy = float((w * dx * dy).sum() / flux_sum)

        # Eigenvalues of covariance (principal axis variances)
        tr = mxx + myy
        det = mxx * myy - mxy * mxy
        disc = max(tr * tr / 4.0 - det, 0.0)
        l1 = tr / 2.0 + np.sqrt(disc)
        l2 = tr / 2.0 - np.sqrt(disc)
        # elongation as sqrt(var_major/var_minor)
        if l2 <= 1e-12:
            elong = float("inf")
        else:
            elong = float(np.sqrt(l1 / l2))

        if elong > float(max_elong):
            continue

        detections.append((peak_sn, flux_sum, cx, cy, elong, npix))

    if not detections:
        return ExtractedSources(
            xy=np.empty((0, 2)),
            shape=full_shape,
            flux=np.empty((0,), dtype=np.float64),
            peak_sn=np.empty((0,), dtype=np.float64),
        )

    # Sort by peak S/N then flux
    detections.sort(key=lambda t: (t[0], t[1]), reverse=True)
    detections = detections[: max_sources if max_sources > 0 else len(detections)]

    peak_sn_arr = np.array([d[0] for d in detections], dtype=np.float64)
    flux_arr = np.array([d[1] for d in detections], dtype=np.float64)
    x = np.array([d[2] for d in detections], dtype=np.float64)
    y = np.array([d[3] for d in detections], dtype=np.float64)

    # Edge margin (in the downsampled/cropped image coords)
    if edge_margin > 0:
        h, w = data.shape
        m = int(edge_margin)
        keep = (x > m) & (y > m) & (x < (w - m)) & (y < (h - m))
        x = x[keep]
        y = y[keep]
        peak_sn_arr = peak_sn_arr[keep]
        flux_arr = flux_arr[keep]

    if x.size == 0:
        return ExtractedSources(
            xy=np.empty((0, 2)),
            shape=full_shape,
            flux=np.empty((0,), dtype=np.float64),
            peak_sn=np.empty((0,), dtype=np.float64),
        )

    # Debug overlay
    if debug_overlay_path is not None:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(data, origin="lower", cmap="gray", vmin=median - 2 * std, vmax=median + 10 * std)
            ax.scatter(x, y, s=18, facecolors="none", edgecolors="lime", linewidths=0.9)
            ax.set_title(f"{fits_path.name}: detections={len(x)} (sn>{sn_thresh}, peak>{peak_sn_thresh})")
            fig.tight_layout()
            fig.savefig(debug_overlay_path, dpi=160)
            plt.close(fig)
        except Exception:
            pass

    # Back to FITS pixel coords (1-based) and undo crop/downsample
    x = (x * downsample) + x0 + 1.0
    y = (y * downsample) + y0 + 1.0
    xy = np.stack([x, y], axis=1)
    return ExtractedSources(xy=xy, shape=full_shape, flux=flux_arr, peak_sn=peak_sn_arr)
