"""WCS construction helpers for assisted plate solving."""

from __future__ import annotations

import numpy as np
from astropy.wcs import WCS


def wcs_from_similarity(
    scale_rad_per_pix: float,
    rotation: np.ndarray,
    translation: np.ndarray,
    ra0_deg: float,
    dec0_deg: float,
) -> WCS:
    cd_rad_per_pix = scale_rad_per_pix * rotation
    cd_deg_per_pix = cd_rad_per_pix * (180.0 / np.pi)

    crpix = -np.linalg.solve(cd_rad_per_pix, translation)

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ("RA---TAN", "DEC--TAN")
    wcs.wcs.crval = [float(ra0_deg), float(dec0_deg)]
    wcs.wcs.crpix = [float(crpix[0]), float(crpix[1])]
    wcs.wcs.cd = cd_deg_per_pix
    return wcs
