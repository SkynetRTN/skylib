"""SkyLib astrometric reduction package.

This module provides a backend-agnostic interface to multiple astrometric
solvers, including Astrometry.net, ASTAP, and PlateSolve 3.8.
"""

from __future__ import absolute_import, division, print_function

from typing import Optional

import numpy

from .backends import available_backends, get_backend
from .backends.astrometry_net import Solver
from .request import SolveRequest
from .solution import Solution
from ..util.angle import angdist

__all__ = ["Solver", "SolveRequest", "Solution", "solve_field", "solve_field_glob"]


def _select_default_backend() -> str:
    preferred = ["an", "astap", "platesolve"]
    available = set(available_backends())
    for name in preferred:
        if name in available:
            return name
    raise RuntimeError("No astrometry backends are available")


def solve_field(
    request: SolveRequest,
    backend: Optional[str] = None,
    engine=None,
) -> Solution:
    """Solve a field using the selected backend."""

    if backend is None:
        backend = _select_default_backend()

    solver_backend = get_backend(backend)
    return solver_backend.solve(request, engine=engine)


def solve_field_glob(
    engine,
    request: SolveRequest,
    min_sources: int = 10,
    initial_radius: float = 1,
    radius_step: float = 0.8,
) -> Solution:
    """
    Obtain astrometric solution given XY coordinates of field stars; works
    for fields around globular clusters.
    """

    if request.xy is None:
        raise ValueError("xy must be provided for solve_field_glob")

    sol = solve_field(request, backend="an", engine=engine)
    if sol.wcs is not None and len(request.xy) >= min_sources:
        n = len(request.xy)
        xy = numpy.asarray(request.xy)
        flux = numpy.asarray(request.flux)
        ra, dec = sol.wcs.all_pix2world(xy[:, 0], xy[:, 1], 1)
        ra %= 360
        ra /= 15
        radius = (dec.max() - dec.min()) / 2
        for ra0, dec0, r0 in engine.globs:
            r = r0 * initial_radius
            found = False
            prev_num_outer = None
            while True:
                inner = angdist(ra0, dec0, ra, dec) < r
                num_inner = inner.sum()
                if not num_inner:
                    # No sources within the current glob radius
                    break

                found = True

                outer = ~inner
                num_outer = n - num_inner
                if num_outer >= min_sources and num_outer != prev_num_outer:
                    # Repeat solution keeping only sources outside the glob
                    prev_num_outer = num_outer
                    new_request = SolveRequest(
                        xy=xy[outer],
                        flux=flux[outer] if request.flux is not None else None,
                        width=request.width,
                        height=request.height,
                        ra_hours=sol.wcs.wcs.crval[0] / 15,
                        dec_degs=sol.wcs.wcs.crval[1],
                        radius=radius,
                        min_scale=request.min_scale,
                        max_scale=request.max_scale,
                        parity=request.parity,
                        sip_order=0,
                        crpix_center=request.crpix_center,
                        max_sources=request.max_sources,
                        retry_lost=False,
                        callback=request.callback,
                        backend_config=request.backend_config,
                    )
                    new_sol = solve_field(new_request, backend="an", engine=engine)
                    if new_sol.wcs is not None:
                        # New solution found with outer sources only
                        sol = new_sol
                        break

                # Not enough stars or solution failed? Decrease the radius.
                r *= radius_step
            if found:
                break
    return sol
