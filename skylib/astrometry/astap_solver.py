"""ASTAP astrometric solver interface.

This module provides a very small wrapper around the `ASTAP` executable.  It
writes the list of detected sources to a temporary FITS image, invokes ASTAP via
``subprocess.run`` and reads the resulting WCS solution using
:class:`astropy.wcs.WCS`.  ASTAP and its star catalog must be installed
separately; this module only points ASTAP to an existing catalog directory.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Iterable, Optional

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from .main import Solution

__all__ = ["solve_astap"]


def _write_fits(xy: np.ndarray, width: int, height: int, path: str) -> None:
    """Write a simple FITS image with point sources at the given coordinates."""
    data = np.zeros((int(height), int(width)), dtype=np.float32)
    for x, y in xy:
        xi = int(round(x)) - 1
        yi = int(round(y)) - 1
        if 0 <= xi < width and 0 <= yi < height:
            data[yi, xi] = 1.0
    fits.writeto(path, data, overwrite=True)


def solve_astap(
    xy: Iterable[Iterable[float]],
    width: int,
    height: int,
    ra_hours: float = 0.0,
    dec_degs: float = 0.0,
    pixel_scale: Optional[float] = None,
    cmd: str = "astap",
    catalog: Optional[str] = None,
) -> Solution:
    """Solve field using ASTAP.

    Parameters
    ----------
    xy : iterable of ``(x, y)``
        Source positions in 1-based pixel coordinates.
    width, height : int
        Image dimensions in pixels.
    ra_hours, dec_degs : float, optional
        Approximate centre of the field.  ``ra_hours`` is specified in hours
        while ``dec_degs`` is in degrees.
    pixel_scale : float, optional
        Pixel scale in arcseconds per pixel.  If omitted ASTAP will attempt to
        determine it automatically.
    cmd : str, optional
        Path to the ASTAP executable.
    catalog : str, optional
        Path to the ASTAP star catalog directory.

    Returns
    -------
    :class:`Solution`
        Astrometric solution object.  If solving fails the ``wcs`` attribute is
        ``None``.
    """
    xy_arr = np.asarray(list(xy))

    with tempfile.TemporaryDirectory() as tmp:
        input_fits = os.path.join(tmp, "field.fits")
        output_fits = os.path.join(tmp, "solved.fits")
        _write_fits(xy_arr, width, height, input_fits)

        cmdline = [cmd, "-f", input_fits, "-o", output_fits,
                   "-r", str(float(ra_hours) * 15.0),
                   "-d", str(float(dec_degs))]
        if pixel_scale is not None:
            cmdline.extend(["-p", str(float(pixel_scale))])
        if catalog:
            cmdline.extend(["-c", catalog])

        subprocess.run(cmdline, check=False)

        sol = Solution()
        if os.path.exists(output_fits):
            try:
                with fits.open(output_fits) as hdul:
                    sol.wcs = WCS(hdul[0].header)
            except Exception:
                sol.wcs = None
        return sol
