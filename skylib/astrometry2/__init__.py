"""Redesigned astrometry module with backend support."""

from .main import (
    AstapBackend,
    AstapConfig,
    AstrometryNetBackend,
    AstrometryNetConfig,
    AstrometryNetSolver,
    Backend,
    PlateSolveBackend,
    PlateSolveConfig,
    SolveRequest,
    SolveSolution,
    solve_field,
    solve_field_glob,
)

__all__ = [
    "AstapBackend",
    "AstapConfig",
    "AstrometryNetBackend",
    "AstrometryNetConfig",
    "AstrometryNetSolver",
    "Backend",
    "PlateSolveBackend",
    "PlateSolveConfig",
    "SolveRequest",
    "SolveSolution",
    "solve_field",
    "solve_field_glob",
]
