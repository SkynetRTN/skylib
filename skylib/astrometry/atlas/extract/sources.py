"""Source extraction utilities for assisted plate solving."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

try:  # pragma: no cover - optional dependency
    from photutils.detection import DAOStarFinder
except Exception:  # pragma: no cover - missing optional dependency
    DAOStarFinder = None


@dataclass
class ExtractedSources:
    xy: np.ndarray
    shape: Tuple[int, int]


def extract_sources(
    fits_path: Path,
    *,
    max_sources: int = 200,
    crop_fraction: float = 0.8,
    downsample: int = 2,
    detection_sigma: float = 0.0,
    fwhm: float = 4.0,
    edge_margin: int = 8,
    # ---- new: quality filters ----
    sharplo: float = 0.2,
    sharphi: float = 0.6,
    roundlo: float = -0.8,
    roundhi: float = 0.8,
    min_snr: float = 0.0,
    peakmax: float | None = None,          # e.g. saturation threshold, if known

    # ---- new: debug overlay ----
    debug_overlay_path: Optional[Path] = None,

) -> ExtractedSources:
    if DAOStarFinder is None:
        raise ImportError("photutils is required for source extraction")

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
    threshold = median + detection_sigma * std

    finder = DAOStarFinder(
        fwhm=fwhm,
        threshold=threshold,
        sharplo=sharplo,
        sharphi=sharphi,
        roundlo=roundlo,
        roundhi=roundhi,
        peakmax=peakmax,  # if None, photutils ignores it
    )

    sources = finder(data - median)
    if sources is None or len(sources) == 0:
        return ExtractedSources(np.empty((0, 2)), full_shape)

    # Compute SNR using the measured flux and the background std
    # (DAOStarFinder returns "flux" already background-subtracted-ish)
    flux = np.asarray(sources["flux"], dtype=np.float64)
    snr = flux / max(float(std), 1e-12)

    # Filter marginal detections by SNR
    keep = snr >= float(min_snr)
    sources = sources[keep]
    if len(sources) == 0:
        return ExtractedSources(np.empty((0, 2)), full_shape)

    # Sort by SNR (often better than raw flux for rejecting junk)
    sources.sort("flux")
    sources.reverse()

    x = np.asarray(sources["xcentroid"], dtype=np.float64)
    y = np.asarray(sources["ycentroid"], dtype=np.float64)

    if edge_margin > 0:
        h, w = data.shape
        mask = (
            (x > edge_margin)
            & (y > edge_margin)
            & (x < (w - edge_margin))
            & (y < (h - edge_margin))
        )
        x = x[mask]
        y = y[mask]

    if len(x) == 0:
        return ExtractedSources(np.empty((0, 2)), full_shape)

    if max_sources > 0 and len(x) > max_sources:
        x = x[:max_sources]
        y = y[:max_sources]

    # --- DEBUG OVERLAY (on the cropped/downsampled image coords) ---
    if debug_overlay_path is not None:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(data, origin="lower", cmap="gray", vmin=median - 2 * std, vmax=median + 10 * std)
            ax.scatter(x, y, s=18, facecolors="none", edgecolors="lime", linewidths=0.9)
            ax.set_title(
                f"{fits_path.name}: detections={len(x)} "
                f"(sigma={detection_sigma}, fwhm={fwhm}, min_snr={min_snr})"
            )
            ax.set_xlabel("x (cropped/downsampled pixels)")
            ax.set_ylabel("y (cropped/downsampled pixels)")
            fig.tight_layout()
            fig.savefig(debug_overlay_path, dpi=160)
            plt.close(fig)
        except Exception:
            # Don't fail plate solving if plotting fails
            pass

    x = (x * downsample) + x0 + 1.0
    y = (y * downsample) + y0 + 1.0

    xy = np.stack([x, y], axis=1)
    return ExtractedSources(xy=xy, shape=full_shape)
