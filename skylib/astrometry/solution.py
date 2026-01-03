"""Shared astrometric solution container."""

from __future__ import annotations

from astropy.wcs import WCS

__all__ = ["Solution"]


class Solution(object):
    """
    Class that encapsulates the results of astrometric reduction, including WCS
    and some solution statistics.

    Attributes::
        wcs: :class:`astropy.wcs.WCS` containing the World Coordinate System
            info for solution; None if solution was not found
        log_odds: logodds of best match
        n_match: number of matched sources
        n_conflict: number of conflicts
        n_field: total number of sources
        index_name: index file name that solved the image
        backend: backend name that produced the solution
        backend_metadata: backend-specific metadata
    """

    wcs: WCS | None = None  # type: WCS
    log_odds: float | None = None
    n_match: int | None = None
    n_conflict: int | None = None
    n_field: int | None = None
    index_name: str | None = None
    backend: str | None = None
    backend_metadata: dict | None = None
