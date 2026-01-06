"""Common astrometry request/solution types and backend protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Protocol

import numpy as np
from astropy.wcs import WCS


@dataclass(frozen=True)
class SolveRequest:
    xy: Optional[np.ndarray] = None
    flux: Optional[np.ndarray] = None
    width: Optional[int] = None
    height: Optional[int] = None
    ra_hours: float = 0.0
    dec_degs: float = 0.0
    radius: float = 180.0
    min_scale: float = 0.1
    max_scale: float = 10.0
    fov: Optional[float] = None
    parity: Optional[bool] = None
    sip_order: int = 3
    crpix_center: bool = True
    max_sources: Optional[int] = None
    retry_lost: bool = True
    callback: Optional[Callable[[], int]] = None
    image_path: Optional[Path] = None
    downsample: Optional[int] = None


@dataclass
class SolveSolution:
    wcs: Optional[WCS] = None
    log_odds: Optional[float] = None
    n_match: Optional[int] = None
    n_conflict: Optional[int] = None
    n_field: Optional[int] = None
    index_name: Optional[str] = None
    backend: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class Backend(Protocol):
    name: str

    def is_available(self) -> bool: ...

    def solve(self, request: SolveRequest, config) -> SolveSolution: ...


__all__ = [
    "Backend",
    "SolveRequest",
    "SolveSolution",
]
