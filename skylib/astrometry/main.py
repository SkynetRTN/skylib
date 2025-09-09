"""SkyLib astrometric reduction package

This module primarily interfaces with the local Astrometry.net engine binding
but can fall back to the `ASTAP <https://www.hnsky.org/astap.htm>`_ solver if
the binding is unavailable or if the user explicitly selects it via
``solve_field(..., backend="astap")``.  The ASTAP executable and its star
catalog must be installed separately; the code only points ASTAP to an existing
catalog.
"""

from __future__ import absolute_import, division, print_function

import os
import sys
from glob import glob
import ctypes
from pathlib import Path

import numpy
from astropy.wcs import Sip, WCS

from skylib.util.angle import angdist

try:  # pragma: no cover - optional dependency
    from . import an_engine
except Exception:  # pragma: no cover - missing optional dependency
    an_engine = None


__all__ = ['Solver', 'solve_field', 'solve_field_glob']


class Solver(object):
    """
    Class that encapsulates the :class:`skylib.astrometry.an_engine.solver_t`
    object and the list of indexes. An instance is created in each solver
    thread and is supplied to :func:`solve_field`.

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
            raise ImportError('an_engine module not available')

        if isinstance(index_path, str):
            index_path = [index_path]

        self.solver = an_engine.solver_new()

        self.indexes = []
        for path in index_path:
            for fn in glob(os.path.join(path, '*')):
                # noinspection PyBroadException
                try:
                    idx = an_engine.index_load(fn, 0, None)
                    if idx is not None:
                        self.indexes.append(idx)
                except Exception:
                    pass

        if not self.indexes:
            raise ValueError('No indexes found')

        # Sort indexes by the number of quads (try smaller indexes first -
        # should be faster)
        self.indexes.sort(key=lambda _idx: _idx.nquads)

        # Load the list of globular clusters
        self.globs = []
        with open(os.path.join(os.path.dirname(__file__), 'ngc2000.dat')) as f:
            for line in f.read().splitlines():
                # noinspection PyBroadException
                try:
                    typ = line[6:9].strip()
                    if typ != 'Gb':
                        continue
                    ra_h, ra_m = line[10:12], line[13:17]
                    dec_s, dec_d, dec_m = line[19], line[20:22], line[23:25]
                    ra = (int(ra_h) + float(ra_m)/60)
                    dec = (1 - 2*(dec_s == '-'))*(int(dec_d) + int(dec_m)/60.0)
                    r = float(line[33:38])/2
                    self.globs.append([ra, dec, r/60])
                except Exception:
                    pass


class Solution(object):
    """
    Class that encapsulates the results of astrometric reduction, including WCS
    and some solution statistics

    Attributes::
        wcs: :class:`astropy.wcs.WCS` containing the World Coordinate System
            info for solution; None if solution was not found
        log_odds: logodds of best match
        n_match: number of matched sources
        n_conflict: number of conflicts
        n_field: total number of sources
        index_name: index file name that solved the image
    """
    wcs: WCS | None = None  # type: WCS
    log_odds: float | None = None
    n_match: int | None = None
    n_conflict: int | None = None
    n_field: int | None = None
    index_name: str | None = None


def array_from_swig(data, shape, dtype=numpy.float64):
    a = numpy.empty(shape, dtype)
    ctypes.memmove(a.ctypes, int(data), a.nbytes)
    return a


def solve_field(engine=None, xy=None, flux=None, width=None, height=None,
                ra_hours=0, dec_degs=0, radius=180, min_scale=0.1,
                max_scale=10, pixel_scale=None, parity=None, sip_order=3,
                crpix_center=True, max_sources=None, retry_lost=True,
                callback=None, backend=None, astap_cmd='astap_cli',
                astap_catalog= "C:/astap", image_path: Path=None) -> Solution:
    """
    Obtain astrometric solution given XY coordinates of field stars

    :param :class:`Solver` engine: Astrometry.net engine solver instance;
        optional when using the ASTAP backend
    :param array_like xy: (n x 2) array of 1-based X and Y pixel coordinates
        of stars
    :param array_like flux: optional n-element array of star fluxes
    :param int width: image width in pixels; defaults to the maximum minus
        minimum X coordinate of stars
    :param int height: image height in pixels; defaults to the maximum minus
        minimum Y coordinate of stars
    :param float ra_hours: optional RA of image center in hours; default: 0
    :param float dec_degs: optional Dec of image center in degrees; default: 0
    :param float radius: optional field search radius in degrees; default: 180
        (search over the whole sky)
    :param float min_scale: optional minimum pixel scale in arcseconds per
        pixel; default: 0.1
    :param float max_scale: optional maximum pixel scale in arcseconds per
        pixel; default: 10
    :param float pixel_scale: pixel scale in arcseconds per pixel for the ASTAP
        backend; ignored for the Astrometry.net backend
    :param bool | None parity: image parity (sign of coordinate transformation
        matrix determinant): True = normal parity, False = flipped image, None
        (default) = try both
    :param int sip_order: order of SIP distortion terms; default: 3;
        0 - disable calculation of distortion
    :param bool crpix_center: set reference pixel to image center
    :param int max_sources: use only the given number of brightest sources;
        0/""/None (default) = no limit
    :param bool retry_lost: if solution failed, retry in the "lost in space"
        mode, i.e. without coordinate restrictions (`radius` = 180) and with
        opposite parity, unless the initial search already had these
        restrictions disabled
    :param callable callback: optional callable that is regularly called
        by the solver, accepts no arguments, and returns 0 to interrupt
        the solution and 1 otherwise
    :param str backend: ``"an"`` for Astrometry.net (default if available) or
        ``"astap"`` to use the ASTAP solver
    :param str astap_cmd: path to ASTAP executable when using the ASTAP backend
    :param str astap_catalog: path to the ASTAP star catalog directory

    :return: astrometric solution object; its `wcs` attribute is set to None if
        solution was not found
    """

    if xy is None:
        raise ValueError('xy must be provided')

    if backend is None:
        backend = 'an' if an_engine is not None else 'astap'

    if backend == 'astap':
        from . import astap_solver
        if image_path is None:
            raise ValueError('image_path must be provided for ASTAP backend')
        
     
        return astap_solver.solve_astap(image_path=image_path, ra_hours=ra_hours,
            dec_degs=dec_degs, radius=radius, pixel_scale=pixel_scale, cmd=astap_cmd,
            catalog=astap_catalog)

    if engine is None:
        raise ValueError('engine must be provided for Astrometry.net backend')
    solver = engine.solver
    ra = float(ra_hours)*15
    dec = float(dec_degs)
    r = float(radius)

    # Set timer callback if requested
    if callback is not None:
        if sys.platform.startswith('win32'):
            time_t = ctypes.c_int64
        elif ctypes.sizeof(ctypes.c_void_p) == ctypes.sizeof(ctypes.c_int64):
            time_t = ctypes.c_int64
        else:
            time_t = ctypes.c_int32
        an_engine.set_timer_callback(
            solver,
            ctypes.cast(
                ctypes.CFUNCTYPE(time_t)(callback),
                ctypes.c_voidp).value)
    else:
        an_engine.set_timer_callback(solver, 0)

    # Set field star position array
    n = len(xy)
    xy = numpy.asanyarray(xy)
    field = an_engine.starxy_new(n, flux is not None, False)
    if flux is not None:
        flux = numpy.asanyarray(flux)
        if len(flux) != n:
            raise ValueError(
                'Flux array must be of the same length as XY array')
        if max_sources:
            order = numpy.argsort(flux)[::-1]
            xy, flux = xy[order], flux[order]
            del order
        an_engine.starxy_set_flux_array(field, flux)
    an_engine.starxy_set_xy_array(field, xy.ravel())
    an_engine.solver_set_field(solver, field)

    try:
        # Initialize solver parameters
        if width:
            minx, maxx = 1, int(width)
        else:
            minx, maxx = xy[:, 0].min(), xy[:, 0].max()
        if height:
            miny, maxy = 1, int(height)
        else:
            miny, maxy = xy[:, 1].min(), xy[:, 1].max()
        an_engine.solver_set_field_bounds(solver, minx, maxx, miny, maxy)
        solver.quadsize_min = 0.1*min(maxx - minx + 1, maxy - miny + 1)

        if crpix_center != '':
            solver.set_crpix = solver.set_crpix_center = int(crpix_center)

        an_engine.solver_set_radec(solver, ra, dec, r)

        solver.funits_lower = float(min_scale)
        solver.funits_upper = float(max_scale)

        solver.logratio_tokeep = numpy.log(1e12)
        solver.distance_from_quad_bonus = True

        if parity is None or parity == '':
            solver.parity = an_engine.PARITY_BOTH
        elif int(parity):
            solver.parity = an_engine.PARITY_NORMAL
        else:
            solver.parity = an_engine.PARITY_FLIP

        enable_sip = sip_order and int(sip_order) >= 2
        if enable_sip:
            solver.do_tweak = True
            solver.tweak_aborder = int(sip_order)
            solver.tweak_abporder = int(sip_order) + 1
        else:
            solver.do_tweak = False

        if max_sources:
            solver.endobj = max_sources
        else:
            solver.endobj = 0

        # Find indexes needed to solve the field
        fmin = solver.quadsize_min*min_scale
        fmax = numpy.hypot(width, height)*max_scale
        indices = []
        for index in engine.indexes:
            if fmin > index.index_scale_upper or \
                    fmax < index.index_scale_lower:
                continue
            if not an_engine.index_is_within_range(index, ra, dec, r):
                continue

            indices.append(index)

        if not len(indices):
            raise ValueError(
                'No indexes found for the given scale and position')

        # Sort indices by scale (larger scales/smaller indices first - should
        # be faster) then by distance from expected position
        indices.sort(
            key=lambda _idx: (
                -_idx.index_scale_upper,
                an_engine.healpix_distance_to_radec(
                    _idx.healpix, _idx.hpnside, ra, dec)[0]
                if _idx.healpix >= 0 else 0,
            ))
        an_engine.solver_clear_indexes(solver)
        for index in indices:
            an_engine.solver_add_index(solver, index)

        # Run the solver
        an_engine.solver_run(solver)
        sol = Solution()

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

            wcs_ctype = ('RA---TAN', 'DEC--TAN')
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
                        a = array_from_swig(
                            sip.a, (maxorder, maxorder)
                        )[:a_order + 1, :a_order + 1]
                        b = array_from_swig(
                            sip.b, (maxorder, maxorder)
                        )[:b_order + 1, :b_order + 1]
                        if a.any() or b.any():
                            ap = array_from_swig(
                                sip.ap, (maxorder, maxorder)
                            )[:ap_order + 1, :ap_order + 1]
                            bp = array_from_swig(
                                sip.bp, (maxorder, maxorder)
                            )[:bp_order + 1, :bp_order + 1]
                            sol.wcs.sip = Sip(
                                a, b, ap, bp,
                                array_from_swig(wcstan.crpix, (2,)))
                            wcs_ctype = ('RA---TAN-SIP', 'DEC--TAN-SIP')
            else:
                wcstan = best_match.wcstan
            sol.wcs.wcs.ctype = wcs_ctype
            sol.wcs.wcs.crpix = array_from_swig(wcstan.crpix, (2,))
            sol.wcs.wcs.crval = array_from_swig(wcstan.crval, (2,))
            sol.wcs.wcs.cd = array_from_swig(wcstan.cd, (2, 2))
        elif retry_lost and (radius < 180 or parity is not None):
            # When no solution was found, retry with all constraints relaxed
            an_engine.solver_cleanup_field(solver)
            return solve_field(
                engine, xy, flux, width, height, 0, 0, 180, min_scale,
                max_scale, None, sip_order, crpix_center, max_sources,
                retry_lost=False)

        return sol
    finally:
        # Cleanup and make solver ready for the next solution
        an_engine.solver_cleanup_field(solver)
        an_engine.solver_clear_indexes(solver)


def solve_field_glob(engine, xy, flux=None, width=None, height=None,
                     ra_hours=0, dec_degs=0, radius=180, min_scale=0.1,
                     max_scale=10, parity=None, sip_order=3, crpix_center=True,
                     max_sources=None, retry_lost=True, callback=None,
                     min_sources=10, initial_radius=1, radius_step=0.8) \
        -> Solution:
    """
    Obtain astrometric solution given XY coordinates of field stars; works
    for fields around globular clusters

    :param :class:`Solver` engine: Astrometry.net engine solver instance
    :param array_like xy: (n x 2) array of 1-based X and Y pixel coordinates
        of stars
    :param array_like flux: optional n-element array of star fluxes
    :param int width: image width in pixels; defaults to the maximum minus
        minimum X coordinate of stars
    :param int height: image height in pixels; defaults to the maximum minus
        minimum Y coordinate of stars
    :param float ra_hours: optional RA of image center in hours; default: 0
    :param float dec_degs: optional Dec of image center in degrees; default: 0
    :param float radius: optional field search radius in degrees; default: 180
        (search over the whole sky)
    :param float min_scale: optional minimum pixel scale in arcseconds per
        pixel; default: 0.1
    :param float max_scale: optional maximum pixel scale in arcseconds per
        pixel; default: 10
    :param bool | None parity: image parity (sign of coordinate transformation
        matrix determinant): True = normal parity, False = flipped image, None
        (default) = try both
    :param int sip_order: order of SIP distortion terms; default: 3;
        0 - disable calculation of distortion
    :param bool crpix_center: set reference pixel to image center
    :param int max_sources: use only the given number of brightest sources;
        0/""/None (default) = no limit
    :param bool retry_lost: if solution failed, retry in the "lost in space"
        mode, i.e. without coordinate restrictions (`radius` = 180) and with
        opposite parity, unless the initial search already had these
        restrictions disabled
    :param callable callback: optional callable that is regularly called
        by the solver, accepts no arguments, and returns 0 to interrupt
        the solution and 1 otherwise
    :param int min_sources: minimum number of sources outside the cluster
        exclusion circle required to find a solution
    :param float initial_radius: initial exclusion circle radius in units
        of the globular cluster radius
    :param float radius_step: gradually decrease the exclusion circle radius
        by this factor if not enough sources outside the circle

    :return: astrometric solution object; its `wcs` attribute is set to None if
        solution was not found
    """
    sol = solve_field(
        engine,
        xy,
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
    if sol.wcs is not None and len(xy) >= min_sources:
        n = len(xy)
        xy = numpy.asarray(xy)
        flux = numpy.asarray(flux)
        ra, dec = sol.wcs.all_pix2world(xy[:, 0], xy[:, 1], 1)
        ra %= 360
        ra /= 15
        radius = (dec.max() - dec.min())/2
        for ra0, dec0, r0 in engine.globs:
            r = r0*initial_radius
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
                    new_sol = solve_field(
                        engine, xy[outer], flux[outer], width, height,
                        sol.wcs.wcs.crval[0]/15, sol.wcs.wcs.crval[1], radius,
                        min_scale, max_scale, parity, 0, crpix_center,
                        max_sources, False, callback)
                    if new_sol.wcs is not None:
                        # New solution found with outer sources only
                        sol = new_sol
                        break

                # Not enough stars or solution failed? Decrease the radius.
                r *= radius_step
            if found:
                break
    return sol
