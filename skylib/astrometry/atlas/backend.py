"""Atlas backend integration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from skylib.astrometry.types import SolveRequest, SolveSolution

from .config import AtlasConfig
from .solve.solver import solve as atlas_solve


class AtlasBackend:
    name = "atlas"

    def is_available(self) -> bool:
        return True

    def solve(self, request: SolveRequest, config: Optional[AtlasConfig]) -> SolveSolution:
        if not isinstance(config, AtlasConfig):
            raise ValueError("Atlas config is required")
        if request.image_path is None:
            raise ValueError("image_path must be provided for Atlas backend")

        catalog_roots = dict(config.catalog_roots) if config.catalog_roots else {}

        if not catalog_roots.get("ucac4"):
            ucac4_root = os.getenv("SKYLIB_UCAC4_ROOT")
            if ucac4_root and Path(ucac4_root).exists():
                catalog_roots["ucac4"] = Path(ucac4_root)

        config.catalog_roots = catalog_roots

        fov_guess = None
        if request.fov is not None:
            fov_guess = (float(request.fov), float(request.fov))

        result = atlas_solve(
            request.image_path,
            config,
            ra0_deg=float(request.ra_hours) * 15.0,
            dec0_deg=float(request.dec_degs),
            scale_range_arcsec_per_pix=(float(request.min_scale), float(request.max_scale)),
            fov_guess_deg=fov_guess,
        )

        sol = SolveSolution(backend=self.name)
        sol.wcs = result.wcs
        sol.metadata = result.metadata
        return sol


__all__ = ["AtlasBackend"]
