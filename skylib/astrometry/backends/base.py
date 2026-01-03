"""Backend interface for astrometric solvers."""

from __future__ import annotations

from typing import Protocol

from ..request import SolveRequest
from ..solution import Solution

__all__ = ["SolverBackend"]


class SolverBackend(Protocol):
    """Interface for backend implementations."""

    name: str

    def is_available(self) -> bool:
        """Return True if the backend can run in this environment."""

    def solve(self, request: SolveRequest, engine=None) -> Solution:
        """Solve an astrometric request."""
