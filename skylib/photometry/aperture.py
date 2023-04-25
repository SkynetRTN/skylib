"""
High-level aperture photometry interface.

:func:`~aperture_photometry()`: fixed or automatic aperture photometry
of an image after source extraction.
"""

from typing import Optional, Union

from numpy import (
    arctan, argsort, array, empty, full_like, indices, int32, isscalar, log,
    log10, ndarray, ndim, ones, pi, sqrt, zeros)
from numpy.lib.recfunctions import append_fields
from numpy.ma import MaskedArray
from scipy.optimize import minimize
import sep

from ..calibration.background import estimate_background, sep_compatible
from ..util.stats import weighted_median


__all__ = ['aperture_photometry']


def calc_flux_err(a: float, img_back: ndarray, x: float, y: float,
                  elongation: float, theta: float, err: ndarray, mask: ndarray,
                  gain: float) -> float:
    """
    Calculate flux error for a source depending on the aperture size; used
    by adaptive photometry to calculate the optimal aperture

    :param a: aperture size
    :param img_back: image with background subtracted
    :param x: source centroid X
    :param y: source centroid Y
    :param elongation: isophotal a/b ratio for the source
    :param theta: major axis position angle in radians
    :param err: background RMS
    :param mask: image mask
    :param gain: inverse gain in e-/ADU

    :return: flux error for the given aperture
    """
    flux, flux_err, flags = sep.sum_ellipse(
        img_back, [x], [y], a, a/elongation, theta, 1, err=err, mask=mask,
        gain=gain, subpix=0)
    if flags:
        raise ValueError('flags = {}'.format(flags))
    if flux <= 0:
        raise ValueError('flux = {}'.format(flux))
    return flux_err/flux


def aperture_photometry(img: Union[ndarray, MaskedArray], sources: ndarray,
                        background: Optional[Union[ndarray,
                                                   MaskedArray]] = None,
                        background_rms: Optional[Union[ndarray,
                                                       MaskedArray]] = None,
                        texp: float = 1,
                        gain: Union[float, ndarray, MaskedArray] = 1,
                        sat_level: float = 63000,
                        a: Optional[float] = None, b: Optional[float] = None,
                        theta: Optional[float] = 0,
                        a_in: Optional[float] = None,
                        a_out: Optional[float] = None,
                        b_out: Optional[float] = None,
                        theta_out: Optional[float] = None,
                        k: float = 0,
                        k_in: Optional[float] = None,
                        k_out: Optional[float] = None,
                        radius: float = 6,
                        fix_aper: bool = False,
                        fix_ell: bool = True,
                        fix_rot: bool = True,
                        apcorr_tol: float = 0.0001) -> ndarray:
    """
    Do automatic or fixed aperture photometry

    :param img: input 2D image array
    :param sources: record array of sources extracted with
        :func:`skylib.extraction.extract_sources`; should contain at least "x"
        and "y" columns
    :param background: optional sky background map; if omitted, extract
        background from the annulus around the aperture, see `a_in` below
    :param background_rms: optional sky background RMS map; if omitted,
        calculate RMS over the annulus around the aperture
    :param texp: exposure time in seconds
    :param gain: electrons to data units conversion factor; used to estimate
        photometric errors; for variable-gain images (e.g. mosaics), must be
        an array of the same shape as the input data
    :param sat_level: saturation level in ADUs; used to select only
        non-saturated stars for adaptive aperture photometry and aperture
        correction
    :param a: fixed aperture radius or semi-major axis in pixels; default: use
        automatic photometry with a = a_iso*k, where a_iso is the isophotal
        semi-major axis
    :param b: semi-minor axis in pixels when using a fixed aperture; default:
        same as `a`
    :param theta: rotation angle of semi-major axis in degrees CCW when using
        a fixed aperture and `b` != `a`; default: 0
    :param a_in: inner annulus radius or semi-major axis in pixels; used
        to estimate the background if `background` or `background_rms` are not
        provided, ignored otherwise; default: `a`*`k_in`
    :param a_out: outer annulus radius or semi-major axis in pixels; default:
        `a`*`k_out`
    :param b_out: outer annulus semi-minor axis in pixels; default: `b`*`k_out`
    :param theta_out: annulus orientation in degrees CCW; default: same
        as `theta`
    :param k: automatic aperture radius in units of isophotal radius; 0 means
        find the optimal radius based on SNR; default: 0
    :param k_in: inner annulus radius in units of aperture radius (fixed
        aperture, i.e. `a` is provided) or isophotal radius (adaptive
        aperture); default: 1.5*`k` or 3.75 if `k` is undefined and `a` = None
    :param k_out: outer annulus radius in units of aperture radius (fixed
        aperture) or isophotal radius (adaptive aperture); default: 2*`k` or
        5 if `k` is undefined and `a` = None
    :param radius: isophotal analysis radius in pixels used to compute
        automatic aperture if ellipse parameters (a,b,theta) are missing
    :param fix_aper: use the same aperture radius for all sources when doing
        automatic photometry; calculated as flux-weighted median of aperture
        sizes based on isophotal parameters
    :param fix_ell: use the same major to minor aperture axis ratio for all
        sources during automatic photometry; calculated as flux-weighted median
        of all ellipticities
    :param fix_rot: use the same aperture position angle for all sources during
        automatic photometry; calculated as flux-weighted median
        of all orientations
    :param apcorr_tol: growth curve stopping tolerance for aperture correction;
        0 = disable aperture correction

    :return: record array containing the input sources, with the following
        fields added or updated: "flux", "flux_err", "mag", "mag_err",
        "aper_a", "aper_b", "aper_theta", "aper_a_in", "aper_a_out",
        "aper_b_out", "aper_theta_out", "aper_area", "background_area",
        "background", "background_rms", "phot_flag"
    """
    if not len(sources):
        return array([])

    img = sep_compatible(img)

    texp = float(texp)
    if isscalar(gain):
        gain = float(gain)
    k = float(k)
    if k <= 0.1:
        k = 0  # temporary fix for k = 0 not being allowed in AgA
    if k_in:
        k_in = float(k_in)
    if k_out:
        k_out = float(k_out)

    x, y = sources['x'] - 1, sources['y'] - 1
    area_img = ones(img.shape, dtype=int32)
    if isinstance(img, MaskedArray):
        mask = img.mask
        img = img.data
    else:
        mask = None

    have_background = background is not None and background_rms is not None
    if have_background:
        background = sep_compatible(background)
        if isinstance(background, MaskedArray):
            if mask is None:
                mask = background.mask
            else:
                mask |= background.mask
            background = background.data

        background_rms = sep_compatible(background_rms)
        if isinstance(background_rms, MaskedArray):
            if mask is None:
                mask = background_rms.mask
            else:
                mask |= background_rms.mask
            background_rms = background_rms.data

    # Will need this to fill the newly added source table columns
    z = zeros(len(sources), float)

    fixed_aper = bool(a)
    if fixed_aper:
        # Use the same fixed aperture and annulus parameters for all sources
        a = float(a)
        if b:
            b = float(b)
        else:
            b = a
        if theta:
            theta = float(theta % 180)*pi/180
            if theta > pi/2:
                theta -= pi
        else:
            theta = 0

        if not have_background:
            if theta_out:
                theta_out = float(theta_out % 180)*pi/180
                if theta_out > pi/2:
                    theta_out -= pi
            elif theta_out != 0:
                theta_out = theta
            if a_in:
                a_in = float(a_in)
            else:
                a_in = a*k_in if k_in else a*1.5*k if k else a*3.75
            if a_out:
                a_out = float(a_out)
            else:
                a_out = a*k_out if k_out else a*2*k if k else a*5
            if b_out:
                b_out = float(b_out)
            else:
                b_out = a_out*b/a
    else:
        # Use automatic apertures derived from ellipse axes; will need image
        # with background subtracted
        if background is None:
            # Estimate background on the fly
            tmp_back, tmp_rms = estimate_background(img, size=64)
        else:
            tmp_back, tmp_rms = background, background_rms
        img_back = img - tmp_back
        for name in ['a', 'b', 'theta', 'flux']:
            if name not in sources.dtype.names:
                sources = append_fields(sources, name, z, usemask=False)
        a, b, theta = sources['a'], sources['b'], sources['theta']
        flux = sources['flux']
        bad = (a <= 0) | (b <= 0) | (flux <= 0)
        if bad.any():
            # Do isophotal analysis to compute ellipse parameters if missing
            yy, xx = indices(img.shape)
            for i in bad.nonzero()[0]:
                ap = (xx - sources[i]['x'])**2 + (yy - sources[i]['y'])**2 <= \
                    radius**2
                if ap.any():
                    yi, xi = ap.nonzero()
                    ap_data = img_back[ap].astype(float)
                    f = ap_data.sum()
                    if f > 0:
                        cx = (xi*ap_data).sum()/f
                        cy = (yi*ap_data).sum()/f
                        x2 = (xi**2*ap_data).sum()/f - cx**2
                        y2 = (yi**2*ap_data).sum()/f - cy**2
                        xy = (xi*yi*ap_data).sum()/f - cx*cy
                    else:
                        cx, cy = xi.mean(), yi.mean()
                        x2 = (xi**2).mean() - cx**2
                        y2 = (yi**2).mean() - cy**2
                        xy = (xi*yi).mean() - cx*cy
                    if x2 == y2:
                        thetai = 0
                    else:
                        thetai = arctan(2*xy/(x2 - y2))/2
                        if y2 > x2:
                            thetai += pi/2
                    m1 = (x2 + y2)/2
                    m2 = sqrt(max((x2 - y2)**2/4 + xy**2, 0))
                    ai = max(1/12, sqrt(max(m1 + m2, 0)))
                    bi = max(1/12, sqrt(max(m1 - m2, 0)))
                    if ai/bi > 2:
                        # Prevent too elongated apertures usually occurring for
                        # faint objects
                        bi = ai
                else:
                    # Cannot obtain a,b,theta from isophotal analysis, assume
                    # circular aperture
                    ai, bi, thetai, f = radius, radius, 0, 0
                a[i] = sources[i]['a'] = ai
                b[i] = sources[i]['b'] = bi
                theta[i] = sources[i]['theta'] = thetai
                flux[i] = sources[i]['flux'] = f
        bad = (a < b).nonzero()
        a[bad], b[bad] = b[bad], a[bad]
        theta[bad] += pi/2
        theta %= pi
        theta[theta > pi/2] -= pi
        elongation = a/b

        # Obtain the optimal aperture radius from the brightest non-saturated
        # source
        if not k:
            for i in argsort(flux)[::-1]:
                if sep.sum_ellipse(
                        img >= sat_level, [x[i]], [y[i]], a[i], b[i], theta[i],
                        1, subpix=0)[0][0]:
                    # Saturated source
                    continue
                try:
                    # noinspection PyTypeChecker
                    res = minimize(
                        calc_flux_err, [a[i]*1.6],
                        (img_back, x[i], y[i], elongation[i], theta[i],
                         tmp_rms, mask, gain), bounds=[(1, None)], tol=1e-5)
                except ValueError:
                    continue
                if not res.success:
                    continue
                k = res.x[0]/a[i]
                break
            if not k:
                raise ValueError(
                    'Not enough data for automatic aperture factor; use '
                    'explicit aperture factor')

        # Calculate weighted median of aperture sizes, elongations, and/or
        # orientations if requested
        r = sqrt(a*b)*k
        if r.size > 1 and any([fix_aper, fix_ell, fix_rot]):
            flux[flux < 0] = 0
            if not flux.any():
                raise ValueError(
                    'Not enough data for weighted median in fixed-aperture '
                    'automatic photometry; use static fixed-aperture or fully '
                    'adaptive automatic photometry instead')
            if fix_aper:
                r = weighted_median(r, flux)
            if fix_ell:
                elongation = weighted_median(elongation, flux)
            if fix_rot:
                theta = weighted_median(theta, flux, period=pi)
                if theta > pi/2:
                    theta -= pi

        # Calculate the final aperture and annulus sizes
        sqrt_el = sqrt(elongation)
        a, b = r*sqrt_el, r/sqrt_el
        if not have_background:
            if not k_in:
                k_in = 1.5*k
            if not k_out:
                k_out = 2*k
            a_in = a*(k_in/k)
            a_out, b_out = a*(k_out/k), b*(k_out/k)
            theta_out = theta

    # Calculate mean and RMS of background; to get the pure sigma, set error
    # to 1 and don't pass the gain
    if have_background:
        if fixed_aper and a == b:
            bk_area = sep.sum_circle(area_img, x, y, a, mask=mask, subpix=0)[0]
            bk_flux = sep.sum_circle(
                background, x, y, a, err=1, mask=mask, subpix=0)[0]
        else:
            bk_area = sep.sum_ellipse(
                area_img, x, y, a, b, theta, 1, mask=mask, subpix=0)[0]
            bk_flux = sep.sum_ellipse(
                background, x, y, a, b, theta, 1, err=1, mask=mask,
                subpix=0)[0]
    elif fixed_aper and a_out == b_out:
        bk_area = sep.sum_circann(
            area_img, x, y, a_in, a_out, mask=mask, subpix=0)[0]
        bk_flux = sep.sum_circann(
            img, x, y, a_in, a_out, mask=mask, subpix=0)[0]
    else:
        bk_area = sep.sum_ellipann(
            area_img, x, y, a_out, b_out, theta_out, a_in/a_out, 1, mask=mask,
            subpix=0)[0]
        bk_flux = sep.sum_ellipann(
            img, x, y, a_out, b_out, theta_out, a_in/a_out, 1, mask=mask,
            subpix=0)[0]

    if have_background:
        area = bk_area
    elif fixed_aper and a == b:
        area = sep.sum_circle(area_img, x, y, a, mask=mask, subpix=0)[0]
    else:
        area = sep.sum_ellipse(
            area_img, x, y, a, b, theta, 1, mask=mask, subpix=0)[0]

    if fixed_aper and a == b:
        # Fixed circular aperture
        if ndim(x) == 1:
            # Separate scalar error for each source
            flux, flux_err = empty([2, len(sources)], dtype=float)
            flags = empty(len(sources), dtype=int)
            if have_background:
                for i, (_x_, _y_) in enumerate(zip(x, y)):
                    flux[i], flux_err[i], flags[i] = sep.sum_circle(
                        img, [_x_], [_y_], a, err=background_rms, mask=mask,
                        gain=gain, subpix=0)
            else:
                for i, (_x_, _y_) in enumerate(zip(x, y)):
                    flux[i], flux_err[i], flags[i] = sep.sum_circle(
                        img, [_x_], [_y_], a, mask=mask, bkgann=(a_in, a_out),
                        gain=gain, subpix=0)
        elif have_background:
            flux, flux_err, flags = sep.sum_circle(
                img, x, y, a, err=background_rms, mask=mask, gain=gain,
                subpix=0)
        else:
            flux, flux_err, flags = sep.sum_circle(
                img, x, y, a, mask=mask, bkgann=(a_in, a_out), gain=gain,
                subpix=0)
    else:
        # Variable or elliptic aperture
        if ndim(x) == 1:
            # Separate scalar error for each source
            flux, flux_err = empty([2, len(sources)], dtype=float)
            flags = empty(len(sources), dtype=int)
            if isscalar(a):
                a = full_like(x, a)
            if isscalar(b):
                b = full_like(x, b)
            if isscalar(theta):
                theta = full_like(x, theta)
            if have_background:
                for i, (_x_, _y_, _a_, _b_, _theta_) in enumerate(zip(
                        x, y, a, b, theta)):
                    flux[i], flux_err[i], flags[i] = sep.sum_ellipse(
                        img, [_x_], [_y_], _a_, _b_, _theta_, 1,
                        err=background_rms, mask=mask, gain=gain, subpix=0)
            else:
                for i, (_x_, _y_, _a_, _b_, _theta_, _a_in_, _a_out_) in \
                        enumerate(zip(x, y, a, b, theta, a_in, a_out)):
                    flux[i], flux_err[i], flags[i] = sep.sum_ellipse(
                        img, [_x_], [_y_], _a_, _b_, _theta_, 1,
                        mask=mask, bkgann=(_a_in_, _a_out_), gain=gain,
                        subpix=0)
        elif have_background:
            flux, flux_err, flags = sep.sum_ellipse(
                img, x, y, a, b, theta, 1, err=background_rms, mask=mask,
                gain=gain, subpix=0)
        else:
            flux, flux_err, flags = sep.sum_ellipse(
                img, x, y, a, b, theta, 1, mask=mask, bkgann=(a_in, a_out),
                gain=gain, subpix=0)

    if have_background:
        # Subtract background from fluxes; background area equals aperture area
        flux -= bk_flux
        bk_mean = bk_flux/area
    else:
        # Background area equals annulus area; background already subtracted
        bk_mean = bk_flux/bk_area

    # Convert ADUs to electrons
    flux *= gain
    flux_err *= gain
    bk_mean *= gain

    # Calculate aperture correction for all aperture sizes from the brightest
    # source
    aper_corr = {}
    if apcorr_tol > 0:
        for i in argsort(flux)[::-1]:
            xi, yi = x[i], y[i]
            if isscalar(a):
                ai = a
            else:
                ai = a[i]
            if isscalar(b):
                bi = b
            else:
                bi = b[i]
            if isscalar(theta):
                thetai = theta
            else:
                thetai = theta[i]

            if ai == bi:
                nsat = sep.sum_circle(
                    img >= sat_level, [xi], [yi], ai, subpix=0)[0][0]
            else:
                nsat = sep.sum_ellipse(
                    img >= sat_level, [xi], [yi], ai, bi, thetai, 1,
                    subpix=0)[0][0]
            if nsat:
                # Saturated source
                continue

            # Obtain total flux by increasing aperture size until it grows
            # either more than before (i.e. a nearby source in the aperture) or
            # less than the threshold (i.e. the growth curve reached
            # saturation)
            f0 = f_prev = flux[i]
            dap = 0
            f_tot = df_prev = None
            while True:
                dap += 0.1
                if ai == bi:
                    f, fl = sep.sum_circle(
                        img, [xi], [yi], ai + dap, mask=mask, subpix=0)[::2]
                    area_i = sep.sum_circle(
                        area_img, [xi], [yi], ai + dap, mask=mask,
                        subpix=0)[0][0]
                else:
                    f, fl = sep.sum_ellipse(
                        img, [xi], [yi], ai + dap, bi*(1 + dap/ai), thetai, 1,
                        mask=mask, subpix=0)[::2]
                    area_i = sep.sum_ellipse(
                        area_img, [xi], [yi], ai + dap, bi + dap, thetai, 1,
                        mask=mask, subpix=0)[0][0]
                f, fl = f[0], fl[0]
                if fl:
                    break
                f = (f - bk_mean[i]*area_i)*gain
                if f <= 0:
                    break
                df = f/f_prev
                if df_prev is not None and df > df_prev:
                    # Increasing growth, nearby source hit
                    f_tot = f_prev
                    break
                if df < 1 + apcorr_tol:
                    # Growth stopped to within the tolerance
                    f_tot = f
                    break
                f_prev, df_prev = f, df
            if f_tot is None:
                continue

            # Calculate fluxes for the chosen source for all unique aperture
            # sizes used for other sources
            fluxes_for_ap = {ai: f0}
            if not isscalar(a):
                for aj in set(a) - {ai}:
                    bj = aj*bi/ai
                    if aj == bj:
                        f, fl = sep.sum_circle(
                            img, [xi], [yi], aj, mask=mask, subpix=0)[::2]
                        area_j = sep.sum_circle(
                            area_img, [xi], [yi], aj, mask=mask,
                            subpix=0)[0][0]
                    else:
                        f, fl = sep.sum_ellipse(
                            img, [xi], [yi], aj, bj, thetai, 1, mask=mask,
                            subpix=0)[::2]
                        area_j = sep.sum_ellipse(
                            area_img, [xi], [yi], aj, bj, thetai, 1,
                            mask=mask, subpix=0)[0][0]
                    f, fl = f[0], fl[0]
                    f = (f - bk_mean[i]*area_j)*gain
                    if fl or f <= 0:
                        continue
                    fluxes_for_ap[aj] = f

            # Calculate aperture corrections
            for aj, f in fluxes_for_ap.items():
                if f < f_tot:
                    aper_corr[aj] = -2.5*log10(f_tot/f)

            break

    if 'flux' in sources.dtype.names:
        sources['flux'] = flux
    else:
        sources = append_fields(sources, 'flux', flux, usemask=False)

    if 'flux_err' in sources.dtype.names:
        sources['flux_err'] = flux_err
    else:
        sources = append_fields(sources, 'flux_err', flux_err, usemask=False)

    if 'flag' in sources.dtype.names:
        sources['flag'] |= flags
    else:
        sources = append_fields(sources, 'flag', flags, usemask=False)

    for name in ['mag', 'mag_err', 'aper_a', 'aper_b', 'aper_theta',
                 'aper_a_in', 'aper_a_out', 'aper_b_out', 'aper_theta_out',
                 'aper_area', 'background_area', 'background',
                 'background_rms']:
        if name not in sources.dtype.names:
            sources = append_fields(sources, name, z, usemask=False)

    good = (flux > 0).nonzero()
    if len(good[0]):
        sources['mag'][good] = -2.5*log10(flux[good]/texp)
        sources['mag_err'][good] = 2.5*log10(1 + flux_err[good]/flux[good])

    sources['aper_a'] = a
    sources['aper_b'] = b
    sources['aper_theta'] = theta

    if not have_background:
        sources['aper_a_in'] = a_in
        sources['aper_a_out'] = a_out
        sources['aper_b_out'] = b_out
        sources['aper_theta_out'] = theta_out

    sources['aper_area'] = area
    sources['background_area'] = bk_area
    sources['background'] = bk_mean

    # Apply aperture correction
    for i in good[0]:
        sources['mag'][i] += aper_corr.get(sources['aper_a'][i], 0)

    return sources
