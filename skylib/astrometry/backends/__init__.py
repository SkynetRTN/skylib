"""Astrometry backend registry."""

from __future__ import annotations

from typing import Dict

from .astap import AstapBackend
from .astrometry_net import AstrometryNetBackend
from .platesolve import PlateSolveBackend

__all__ = ["available_backends", "get_backend"]


_BACKENDS: Dict[str, object] = {
    "an": AstrometryNetBackend(),
    "astrometry_net": AstrometryNetBackend(),
    "astap": AstapBackend(),
    "platesolve": PlateSolveBackend(),
}


def get_backend(name: str):
    try:
        return _BACKENDS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown backend '{name}'") from exc


def available_backends() -> list[str]:
    return [name for name, backend in _BACKENDS.items() if backend.is_available()]
