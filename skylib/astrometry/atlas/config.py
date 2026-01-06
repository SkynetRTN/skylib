"""Configuration for the Atlas plate solver."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional


@dataclass
class AtlasConfig:
    catalog: str = "ucac5"
    catalog_roots: Mapping[str, Path] = field(default_factory=dict)
    timeout_s: Optional[float] = None
    max_catalog_stars: int = 10000
    max_image_stars: int = 500
    n_tri_obs: int = 15000
    n_tri_cat: int = 50000
    invariant_tol: float = 0.006
    match_tol_arcsec: float = 5.0
    refine_center: bool = True
    thin: int = 1
    debug: bool = False
    catalog_pad_frac = 1.0  # +50% radius
    catalog_max_radius_deg = None  # or e.g. 2.0

    def resolve_catalog(self) -> tuple[str, Path]:
        catalog = self.catalog.strip().lower()
        if catalog in self.catalog_roots:
            return catalog, self.catalog_roots[catalog]
        raise ValueError(f"Unsupported catalog: {self.catalog}")
