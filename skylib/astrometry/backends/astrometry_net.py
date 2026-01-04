"""Astrometry.net backend implementation."""

from __future__ import annotations

import ctypes
import importlib
import os
import sys
from glob import glob

import numpy
from astropy.wcs import Sip, WCS

from ..request import SolveRequest
from ..solution import Solution

_an_engine_spec = importlib.util.find_spec("skylib.astrometry.an_engine")
if _an_engine_spec is None:  # pragma: no cover - optional dependency
    an_engine = None
else:  # pragma: no cover - optional dependency
    an_engine = importlib.import_module("skylib.astrometry.an_engine")

__all__ = ["AstrometryNetBackend", "Solver"]


class Solver(object):
    """
    Class that encapsulates the :class:`skylib.astrometry.an_engine.solver_t`
    object and the list of indexes. An instance is created in each solver
    thread and is supplied to the astrometry.net backend.

    Attributes::
        solver: Astrometry.net engine :class:`an_engine.solver_t` object
        indexes: list of :class:`an_engine.index_t` instances
        globs: (RA_hours, Dec_degs, r_arcmins) list of globular clusters
            utilized by :func:`solve_field_glob` to find a WCS solution
            in the vicinity of a globular cluster
    """

    globs = None  # type: list

    def __init__(self, index_path):
        """
        Create solver

        :param str | list index_path: directory or list of directories
            containing index files
        """
        if an_engine is None:
            raise ImportError("an_engine module not available")

        if isinstance(index_path, str):
            index_path = [index_path]

        self.solver = an_engine.solver_new()

        self.indexes = []
        for path in index_path:
            for fn in glob(os.path.join(path, "*")):
                # noinspection PyBroadException
                try:
                    idx = an_engine.index_load(fn, 0, None)
                    if idx is not None:
                        self.indexes.append(idx)
                except Exception:
                    pass

        if not self.indexes:
            raise ValueError("No indexes found")

        # Sort indexes by the number of quads (try smaller indexes first -
        # should be faster)
        self.indexes.sort(key=lambda _idx: _idx.nquads)

        # Load the list of globular clusters
        self.globs = []
        with open(os.path.join(os.path.dirname(__file__), "..", "ngc2000.dat")) as f:
            for line in f.read().splitlines():
                # noinspection PyBroadException
                try:
                    typ = line[6:9].strip()
                    if typ != "Gb":
                        continue
                    ra_h, ra_m = line[10:12], line[13:17]
                    dec_s, dec_d, dec_m = line[19], line[20:22], line[23:25]
                    ra = (int(ra_h) + float(ra_m) / 60)
                    dec = (1 - 2 * (dec_s == "-")) * (
                        int(dec_d) + int(dec_m) / 60.0
                    )
                    r = float(line[33:38]) / 2
                    self.globs.append([ra, dec, r / 60])
                except Exception:
                    pass


def _array_from_swig(data, shape, dtype=numpy.float64):
    a = numpy.empty(shape, dtype)
    ctypes.memmove(a.ctypes, int(data), a.nbytes)
    return a


class AstrometryNetBackend:
    name = "an"

    def is_available(self) -> bool:
        return an_engine is not None

    def solve(self, request: SolveRequest, engine=None) -> Solution:
        if an_engine is None:
            raise ImportError("an_engine module not available")

        if request.xy is None:
            raise ValueError("xy must be provided for Astrometry.net backend")

        if engine is None:
            raise ValueError("engine must be provided for Astrometry.net backend")

        solver = engine.solver
        ra = float(request.ra_hours) * 15
        dec = float(request.dec_degs)
        r = float(request.radius)

        # Set timer callback if requested
        if request.callback is not None:
            if sys.platform.startswith("win32"):
                time_t = ctypes.c_int64
            elif ctypes.sizeof(ctypes.c_void_p) == ctypes.sizeof(ctypes.c_int64):
                time_t = ctypes.c_int64
            else:
                time_t = ctypes.c_int32
            an_engine.set_timer_callback(
                solver,
                ctypes.cast(ctypes.CFUNCTYPE(time_t)(request.callback), ctypes.c_voidp)
                .value,
            )
        else:
            an_engine.set_timer_callback(solver, 0)

        # Set field star position array
        xy = numpy.asanyarray(request.xy)
        n = len(xy)
        flux = request.flux
        field = an_engine.starxy_new(n, flux is not None, False)
        if flux is not None:
            flux = numpy.asanyarray(flux)
            if len(flux) != n:
                raise ValueError("Flux array must be of the same length as XY array")
            if request.max_sources:
                order = numpy.argsort(flux)[::-1]
                xy, flux = xy[order], flux[order]
                del order
            an_engine.starxy_set_flux_array(field, flux)
        an_engine.starxy_set_xy_array(field, xy.ravel())
        an_engine.solver_set_field(solver, field)

        try:
            # Initialize solver parameters
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

            solver.logratio_tokeep = numpy.log(1e12)
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

            # Find indexes needed to solve the field
            fmin = solver.quadsize_min * request.min_scale
            fmax = numpy.hypot(request.width, request.height) * request.max_scale
            indices = []
            for index in engine.indexes:
                if fmin > index.index_scale_upper or fmax < index.index_scale_lower:
                    continue
                if not an_engine.index_is_within_range(index, ra, dec, r):
                    continue

                indices.append(index)

            if not len(indices):
                raise ValueError("No indexes found for the given scale and position")

            # Sort indices by scale (larger scales/smaller indices first - should
            # be faster) then by distance from expected position
            indices.sort(
                key=lambda _idx: (
                    -_idx.index_scale_upper,
                    an_engine.healpix_distance_to_radec(
                        _idx.healpix, _idx.hpnside, ra, dec
                    )[0]
                    if _idx.healpix >= 0
                    else 0,
                )
            )
            an_engine.solver_clear_indexes(solver)
            for index in indices:
                an_engine.solver_add_index(solver, index)

            # Run the solver
            an_engine.solver_run(solver)
            sol = Solution()
            sol.backend = self.name

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
                # Get WCS parameters of best solution
                sol.wcs = WCS(naxis=2)

                wcs_ctype = ("RA---TAN", "DEC--TAN")
                if enable_sip:
                    sip = best_match.sip
                    try:
                        wcstan = sip.wcstan
                    except AttributeError:
                        # Unable to compute SIP distortions, too few sources?
                        wcstan = best_match.wcstan
                    else:
                        a_order, b_order = sip.a_order, sip.b_order
                        if a_order > 0 or b_order > 0:
                            ap_order, bp_order = sip.ap_order, sip.bp_order
                            maxorder = an_engine.SIP_MAXORDER
                            a = _array_from_swig(
                                sip.a, (maxorder, maxorder)
                            )[: a_order + 1, : a_order + 1]
                            b = _array_from_swig(
                                sip.b, (maxorder, maxorder)
                            )[: b_order + 1, : b_order + 1]
                            if a.any() or b.any():
                                ap = _array_from_swig(
                                    sip.ap, (maxorder, maxorder)
                                )[: ap_order + 1, : ap_order + 1]
                                bp = _array_from_swig(
                                    sip.bp, (maxorder, maxorder)
                                )[: bp_order + 1, : bp_order + 1]
                                sol.wcs.sip = Sip(
                                    a, b, ap, bp, _array_from_swig(wcstan.crpix, (2,))
                                )
                                wcs_ctype = ("RA---TAN-SIP", "DEC--TAN-SIP")
                else:
                    wcstan = best_match.wcstan
                sol.wcs.wcs.ctype = wcs_ctype
                sol.wcs.wcs.crpix = _array_from_swig(wcstan.crpix, (2,))
                sol.wcs.wcs.crval = _array_from_swig(wcstan.crval, (2,))
                sol.wcs.wcs.cd = _array_from_swig(wcstan.cd, (2, 2))
            elif request.retry_lost and (request.radius < 180 or request.parity is not None):
                # When no solution was found, retry with all constraints relaxed
                an_engine.solver_cleanup_field(solver)
                relaxed_request = SolveRequest(
                    xy=request.xy,
                    flux=request.flux,
                    width=request.width,
                    height=request.height,
                    ra_hours=0,
                    dec_degs=0,
                    radius=180,
                    min_scale=request.min_scale,
                    max_scale=request.max_scale,
                    parity=None,
                    sip_order=request.sip_order,
                    crpix_center=request.crpix_center,
                    max_sources=request.max_sources,
                    retry_lost=False,
                    callback=request.callback,
                    backend_config=request.backend_config,
                )
                return self.solve(relaxed_request, engine=engine)

            return sol
        finally:
            # Cleanup and make solver ready for the next solution
            an_engine.solver_cleanup_field(solver)
            an_engine.solver_clear_indexes(solver)
