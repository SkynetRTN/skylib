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
    max_image_stars: int = 120
    n_tri_obs: int = 8000
    n_tri_cat: int = 50000
    invariant_tol: float = 0.006
    match_tol_arcsec: float = 3.0
    refine_center: bool = True
    thin: int = 1
    debug: bool = False

    def resolve_catalog(self) -> tuple[str, Path]:
        catalog = self.catalog.strip().lower()
        if catalog in self.catalog_roots:
            return catalog, self.catalog_roots[catalog]
        raise ValueError(f"Unsupported catalog: {self.catalog}")
