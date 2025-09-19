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
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from .main import Solution

__all__ = ["solve_astap"]



def solve_astap(
    image_path: Path,
    ra_hours: float = 0.0,
    dec_degs: float = 0.0,
    fov: Optional[float] = None,
    radius: float = 1.0,
    cmd: str = "astap_cli",
    catalog: Optional[str] = "C:/astap",
    downsample: Optional[int] = None,
) -> Solution:
    """Solve field using ASTAP.

    Parameters
    ----------
    image_path : str
        Path to the FITS image file containing the field to be solved.
    ra_hours, dec_degs : float, optional
        Approximate centre of the field.  ``ra_hours`` is specified in hours
        while ``dec_degs`` is in degrees.
    fov : float, optional
        Field of view in degrees.  If omitted ASTAP will attempt to determine it
        automatically.
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
    
    with tempfile.TemporaryDirectory() as tmp:
        output = os.path.join(tmp, "solved")

        cmdline = [cmd, "-f", str(image_path), "-o", output]
        if ra_hours is not None:
            cmdline.extend(["-ra", str(float(ra_hours))])

        if dec_degs is not None:
            cmdline.extend(["-spd", str(float(dec_degs+90.))])
        
        if radius is not None:
            cmdline.extend(["-r", str(float(radius))])

        if fov is not None:
            cmdline.extend(["-fov", str(float(fov))])
            
        if catalog:
            cmdline.extend(["-d", catalog])
        
        if downsample is not None:
            cmdline.extend(["-z", str(int(downsample))])

        print("Running: ", " ".join(cmdline))
        subprocess.run(cmdline, check=False)

        sol = Solution()
        output = output + ".wcs"
        if os.path.exists(output):
            try:
                hdr = fits.Header.fromtextfile(output)
                sol.wcs = WCS(hdr)
            except Exception as e:
                print("Failed to read WCS:", e)
                sol.wcs = None
        return sol
