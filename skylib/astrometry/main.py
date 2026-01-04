"""SkyLib astrometric reduction package.

This legacy module now forwards solving requests to ``skylib.astrometry2``
while preserving the original ``solve_field`` and ``solve_field_glob``
interfaces.
"""

from __future__ import absolute_import, division, print_function

from pathlib import Path
from typing import Optional

from skylib.astrometry2 import (
    AstapConfig,
    AstrometryNetConfig,
    AstrometryNetSolver,
    SolveRequest,
    SolveSolution,
    solve_field as solve_field_new,
    solve_field_glob as solve_field_glob_new,
)

try:  # pragma: no cover - optional dependency
    from skylib.astrometry2 import an_engine
except Exception:  # pragma: no cover - missing optional dependency
    an_engine = None

__all__ = ["Solver", "solve_field", "solve_field_glob", "Solution"]

Solver = AstrometryNetSolver
Solution = SolveSolution


def solve_field(
    engine=None,
    xy=None,
    flux=None,
    width=None,
    height=None,
    ra_hours=0,
    dec_degs=0,
    radius=180,
    min_scale=0.1,
    max_scale=10,
    fov=None,
    parity=None,
    sip_order=3,
    crpix_center=True,
    max_sources=None,
    retry_lost=True,
    callback=None,
    backend=None,
    astap_cmd="astap_cli",
    astap_catalog="C:/astap",
    image_path: Path = None,
    downsample: Optional[int] = None,
) -> SolveSolution:
    """Obtain astrometric solution given XY coordinates of field stars."""

    if backend is None:
        backend = "an" if an_engine is not None else "astap"
    elif backend in {"an", "astrometry.net"} and an_engine is None:
        raise ValueError("Astrometry.net backend is not available on this system")

    request = SolveRequest(
        xy=xy,
        flux=flux,
        width=width,
        height=height,
        ra_hours=ra_hours,
        dec_degs=dec_degs,
        radius=radius,
        min_scale=min_scale,
        max_scale=max_scale,
        fov=fov,
        parity=parity,
        sip_order=sip_order,
        crpix_center=crpix_center,
        max_sources=max_sources,
        retry_lost=retry_lost,
        callback=callback,
        image_path=image_path,
        downsample=downsample,
    )

    configs = {}
    if backend in {"an", "astrometry.net"}:
        if engine is None:
            raise ValueError("engine must be provided for Astrometry.net backend")
        configs["an"] = AstrometryNetConfig(engine=engine)
    elif backend == "astap":
        configs["astap"] = AstapConfig(cmd=astap_cmd, catalog=astap_catalog)

    return solve_field_new(request, backend=backend, configs=configs)


def solve_field_glob(
    engine,
    xy,
    flux=None,
    width=None,
    height=None,
    ra_hours=0,
    dec_degs=0,
    radius=180,
    min_scale=0.1,
    max_scale=10,
    parity=None,
    sip_order=3,
    crpix_center=True,
    max_sources=None,
    retry_lost=True,
    callback=None,
    min_sources=10,
    initial_radius=1,
    radius_step=0.8,
) -> SolveSolution:
    """Obtain astrometric solution for fields around globular clusters."""

    request = SolveRequest(
        xy=xy,
        flux=flux,
        width=width,
        height=height,
        ra_hours=ra_hours,
        dec_degs=dec_degs,
        radius=radius,
        min_scale=min_scale,
        max_scale=max_scale,
        parity=parity,
        sip_order=sip_order,
        crpix_center=crpix_center,
        max_sources=max_sources,
        retry_lost=retry_lost,
        callback=callback,
    )

    config = AstrometryNetConfig(engine=engine)
    return solve_field_glob_new(
        request,
        config,
        min_sources=min_sources,
        initial_radius=initial_radius,
        radius_step=radius_step,
    )
