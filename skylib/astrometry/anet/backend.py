"""Astrometry.net backend integration."""

from __future__ import annotations

import ctypes
import sys
from typing import Optional

import numpy as np
from astropy.wcs import Sip, WCS

from skylib.util.angle import angdist

from ..types import SolveRequest, SolveSolution
from .config import AstrometryNetConfig
from .engine import AstrometryNetSolver, an_engine


class AstrometryNetBackend:
    name = "an"

    def is_available(self) -> bool:
        return an_engine is not None

    def solve(
        self,
        request: SolveRequest,
        config: Optional[AstrometryNetConfig],
    ) -> SolveSolution:
        if request.xy is None:
            raise ValueError("xy must be provided")

        if not isinstance(config, AstrometryNetConfig):
            raise ValueError("Astrometry.net config is required")

        engine = config.engine
        if engine is None:
            if config.index_path is None:
                raise ValueError("index_path or engine must be provided")
            engine = AstrometryNetSolver(config.index_path)

        solver = engine.solver
        ra = float(request.ra_hours) * 15
        dec = float(request.dec_degs)
        r = float(request.radius)

        if request.callback is not None:
            if sys.platform.startswith("win32"):
                time_t = ctypes.c_int64
            elif ctypes.sizeof(ctypes.c_void_p) == ctypes.sizeof(ctypes.c_int64):
                time_t = ctypes.c_int64
            else:
                time_t = ctypes.c_int32
            an_engine.set_timer_callback(
                solver,
                ctypes.cast(ctypes.CFUNCTYPE(time_t)(request.callback), ctypes.c_voidp).value,
            )
        else:
            an_engine.set_timer_callback(solver, 0)

        n = len(request.xy)
        xy = np.asanyarray(request.xy)
        field = an_engine.starxy_new(n, request.flux is not None, False)
        if request.flux is not None:
            flux = np.asanyarray(request.flux)
            if len(flux) != n:
                raise ValueError("Flux array must be of the same length as XY array")
            if request.max_sources:
                order = np.argsort(flux)[::-1]
                xy, flux = xy[order], flux[order]
                del order
            an_engine.starxy_set_flux_array(field, flux)
        an_engine.starxy_set_xy_array(field, xy.ravel())
        an_engine.solver_set_field(solver, field)

        try:
            if request.width:
                minx, maxx = 1, int(request.width)
            else:
                minx, maxx = xy[:, 0].min(), xy[:, 0].max()
            if request.height:
                miny, maxy = 1, int(request.height)
            else:
                miny, maxy = xy[:, 1].min(), xy[:, 1].max()
            an_engine.solver_set_field_bounds(solver, minx, maxx, miny, maxy)
            solver.quadsize_min = 0.1 * min(maxx - minx + 1, maxy - miny + 1)

            if request.crpix_center != "":
                solver.set_crpix = solver.set_crpix_center = int(request.crpix_center)

            an_engine.solver_set_radec(solver, ra, dec, r)

            solver.funits_lower = float(request.min_scale)
            solver.funits_upper = float(request.max_scale)

            solver.logratio_tokeep = np.log(1e12)
            solver.distance_from_quad_bonus = True

            if request.parity is None or request.parity == "":
                solver.parity = an_engine.PARITY_BOTH
            elif int(request.parity):
                solver.parity = an_engine.PARITY_NORMAL
            else:
                solver.parity = an_engine.PARITY_FLIP

            enable_sip = request.sip_order and int(request.sip_order) >= 2
            if enable_sip:
                solver.do_tweak = True
                solver.tweak_aborder = int(request.sip_order)
                solver.tweak_abporder = int(request.sip_order) + 1
            else:
                solver.do_tweak = False

            if request.max_sources:
                solver.endobj = request.max_sources
            else:
                solver.endobj = 0

            if request.width is None or request.height is None:
                width = maxx - minx + 1
                height = maxy - miny + 1
            else:
                width = request.width
                height = request.height

            fmin = solver.quadsize_min * request.min_scale
            fmax = np.hypot(width, height) * request.max_scale
            indices = []
            for index in engine.indexes:
                if fmin > index.index_scale_upper or fmax < index.index_scale_lower:
                    continue
                if not an_engine.index_is_within_range(index, ra, dec, r):
                    continue

                indices.append(index)

            if not indices:
                raise ValueError("No indexes found for the given scale and position")

            indices.sort(
                key=lambda _idx: (
                    -_idx.index_scale_upper,
                    an_engine.healpix_distance_to_radec(
                        _idx.healpix,
                        _idx.hpnside,
                        ra,
                        dec,
                    )[0]
                    if _idx.healpix >= 0
                    else 0,
                )
            )
            an_engine.solver_clear_indexes(solver)
            for index in indices:
                an_engine.solver_add_index(solver, index)

            an_engine.solver_run(solver)
            sol = SolveSolution(backend=self.name)

            if solver.have_best_match:
                best_match = solver.best_match
                sol.log_odds = solver.best_logodds
                sol.n_match = best_match.nmatch
                sol.n_conflict = best_match.nconflict
                sol.n_field = best_match.nfield
                if best_match.index is not None:
                    sol.index_name = best_match.index.indexname
            else:
                best_match = None

            if solver.best_match_solves:
                sol.wcs = WCS(naxis=2)

                wcs_ctype = ("RA---TAN", "DEC--TAN")
                if enable_sip:
                    sip = best_match.sip
                    try:
                        wcstan = sip.wcstan
                    except AttributeError:
                        wcstan = best_match.wcstan
                    else:
                        a_order, b_order = sip.a_order, sip.b_order
                        if a_order > 0 or b_order > 0:
                            ap_order, bp_order = sip.ap_order, sip.bp_order
                            maxorder = an_engine.SIP_MAXORDER
                            a = array_from_swig(sip.a, (maxorder, maxorder))[
                                : a_order + 1, : a_order + 1
                            ]
                            b = array_from_swig(sip.b, (maxorder, maxorder))[
                                : b_order + 1, : b_order + 1
                            ]
                            if a.any() or b.any():
                                ap = array_from_swig(sip.ap, (maxorder, maxorder))[
                                    : ap_order + 1,
                                    : ap_order + 1,
                                ]
                                bp = array_from_swig(sip.bp, (maxorder, maxorder))[
                                    : bp_order + 1,
                                    : bp_order + 1,
                                ]
                                sol.wcs.sip = Sip(
                                    a,
                                    b,
                                    ap,
                                    bp,
                                    array_from_swig(wcstan.crpix, (2,)),
                                )
                                wcs_ctype = ("RA---TAN-SIP", "DEC--TAN-SIP")
                else:
                    wcstan = best_match.wcstan
                sol.wcs.wcs.ctype = wcs_ctype
                sol.wcs.wcs.crpix = array_from_swig(wcstan.crpix, (2,))
                sol.wcs.wcs.crval = array_from_swig(wcstan.crval, (2,))
                sol.wcs.wcs.cd = array_from_swig(wcstan.cd, (2, 2))
            elif request.retry_lost and (request.radius < 180 or request.parity is not None):
                an_engine.solver_cleanup_field(solver)
                retry_request = SolveRequest(
                    xy=request.xy,
                    flux=request.flux,
                    width=request.width,
                    height=request.height,
                    ra_hours=0,
                    dec_degs=0,
                    radius=180,
                    min_scale=request.min_scale,
                    max_scale=request.max_scale,
                    fov=None,
                    parity=None,
                    sip_order=request.sip_order,
                    crpix_center=request.crpix_center,
                    max_sources=request.max_sources,
                    retry_lost=False,
                    callback=request.callback,
                )
                return self.solve(retry_request, config)

            return sol
        finally:
            an_engine.solver_cleanup_field(solver)
            an_engine.solver_clear_indexes(solver)


def solve_field_glob(
    request: SolveRequest,
    config: AstrometryNetConfig,
    min_sources: int = 10,
    initial_radius: float = 1,
    radius_step: float = 0.8,
) -> SolveSolution:
    backend = AstrometryNetBackend()
    sol = backend.solve(request, config)
    engine = config.engine
    if engine is None:
        if config.index_path is None:
            raise ValueError("index_path or engine must be provided")
        engine = AstrometryNetSolver(config.index_path)

    if sol.wcs is not None and request.xy is not None and len(request.xy) >= min_sources:
        n = len(request.xy)
        xy = np.asarray(request.xy)
        flux = np.asarray(request.flux) if request.flux is not None else np.zeros(n)
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
                    break

                found = True

                outer = ~inner
                num_outer = n - num_inner
                if num_outer >= min_sources and num_outer != prev_num_outer:
                    prev_num_outer = num_outer
                    new_request = SolveRequest(
                        xy=xy[outer],
                        flux=flux[outer],
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
                    )
                    new_sol = backend.solve(new_request, config)
                    if new_sol.wcs is not None:
                        sol = new_sol
                        break

                r *= radius_step
            if found:
                break
    return sol


def array_from_swig(data, shape, dtype=np.float64):
    a = np.empty(shape, dtype)
    ctypes.memmove(a.ctypes, int(data), a.nbytes)
    return a


__all__ = [
    "AstrometryNetBackend",
    "solve_field_glob",
]
