"""Backend configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

__all__ = [
    "BackendConfig",
    "AstapConfig",
    "PlateSolveConfig",
]


class BackendConfig:
    """Marker base class for backend configurations."""


@dataclass
class AstapConfig(BackendConfig):
    cmd: str = "astap_cli"
    catalog: str | None = "C:/astap"
    downsample: int | None = None


@dataclass
class PlateSolveConfig(BackendConfig):
    cmd: str | None = None
    cmdline: Sequence[str] | None = None
    wcs_path: Path | None = None
