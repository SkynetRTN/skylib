"""Backend-agnostic solver request definition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy

from .configs import BackendConfig

__all__ = ["SolveRequest"]


@dataclass
class SolveRequest:
    """Backend-agnostic request for astrometric solving."""

    image_path: Optional[Path] = None
    xy: Optional[numpy.ndarray] = None
    flux: Optional[numpy.ndarray] = None
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
    callback: Optional[callable] = None
    downsample: Optional[int] = None
    backend_config: Optional[BackendConfig] = None
