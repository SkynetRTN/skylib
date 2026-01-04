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

try:  # pragma: no cover - optional dependency
    from . import an_engine
except Exception:  # pragma: no cover - missing optional dependency
    an_engine = None

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
    "an_engine",
]
