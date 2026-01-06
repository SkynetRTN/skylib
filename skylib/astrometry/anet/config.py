"""Astrometry.net backend configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .engine import AstrometryNetSolver


@dataclass
class AstrometryNetConfig:
    index_path: Optional[Union[str, Sequence[str]]] = None
    engine: Optional["AstrometryNetSolver"] = None


__all__ = ["AstrometryNetConfig"]
