"""ASTAP backend implementation."""

from __future__ import annotations

from pathlib import Path

from ..astap_solver import solve_astap
from ..configs import AstapConfig
from ..request import SolveRequest
from ..solution import Solution

__all__ = ["AstapBackend"]


class AstapBackend:
    name = "astap"

    def is_available(self) -> bool:
        return True

    def solve(self, request: SolveRequest, engine=None) -> Solution:
        if request.image_path is None:
            raise ValueError("image_path must be provided for ASTAP backend")

        if request.backend_config is None:
            config = AstapConfig()
        elif isinstance(request.backend_config, AstapConfig):
            config = request.backend_config
        else:
            raise TypeError("ASTAP backend requires AstapConfig configuration")

        return solve_astap(
            image_path=Path(request.image_path),
            ra_hours=request.ra_hours,
            dec_degs=request.dec_degs,
            radius=request.radius,
            fov=request.fov,
            cmd=config.cmd,
            catalog=config.catalog,
            downsample=(
                config.downsample
                if config.downsample is not None
                else request.downsample
            ),
        )
