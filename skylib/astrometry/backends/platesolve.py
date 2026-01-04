"""PlateSolve 3.8 backend implementation."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from astropy.io import fits
from astropy.wcs import WCS

from ..configs import PlateSolveConfig
from ..request import SolveRequest
from ..solution import Solution

__all__ = ["PlateSolveBackend"]


class PlateSolveBackend:
    name = "platesolve"

    def is_available(self) -> bool:
        return True

    def _build_cmdline(self, request: SolveRequest) -> list[str]:
        if request.backend_config is None:
            config = PlateSolveConfig()
        elif isinstance(request.backend_config, PlateSolveConfig):
            config = request.backend_config
        else:
            raise TypeError("PlateSolve backend requires PlateSolveConfig configuration")

        cmdline = config.cmdline
        if cmdline:
            return [str(part).format(
                image_path=request.image_path,
                ra_hours=request.ra_hours,
                dec_degs=request.dec_degs,
                fov=request.fov,
                radius=request.radius,
            ) for part in cmdline]

        cmd = config.cmd
        if not cmd:
            raise ValueError(
                "PlateSolve backend requires PlateSolveConfig.cmd or "
                "PlateSolveConfig.cmdline"
            )

        if request.image_path is None:
            raise ValueError("image_path must be provided for PlateSolve backend")

        return [cmd, str(request.image_path)]

    def solve(self, request: SolveRequest, engine=None) -> Solution:
        if request.image_path is None:
            raise ValueError("image_path must be provided for PlateSolve backend")

        cmdline = self._build_cmdline(request)
        subprocess.run(cmdline, check=False)

        if request.backend_config is None:
            config = PlateSolveConfig()
        elif isinstance(request.backend_config, PlateSolveConfig):
            config = request.backend_config
        else:
            raise TypeError("PlateSolve backend requires PlateSolveConfig configuration")

        wcs_path = config.wcs_path
        if wcs_path is None:
            wcs_path = Path(request.image_path).with_suffix(".wcs")
        else:
            wcs_path = Path(wcs_path)

        sol = Solution()
        sol.backend = self.name
        sol.backend_metadata = {"cmdline": cmdline}

        if os.path.exists(wcs_path):
            try:
                hdr = fits.Header.fromtextfile(str(wcs_path))
                sol.wcs = WCS(hdr)
            except Exception:
                sol.wcs = None
        return sol
