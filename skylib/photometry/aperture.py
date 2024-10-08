"""
Aperture photometry.

:func:`~aperture_photometry()`: fixed or automatic aperture photometry of an image after source extraction.
"""

import numpy as np
from numpy.lib.recfunctions import append_fields
from numpy.ma import MaskedArray
from scipy.optimize import minimize
from numba import njit, prange

from ..calibration.background import estimate_background, sep_compatible
from ..util.stats import weighted_median
from .aperture_numba import sum_circle, sum_ellipse, sum_circann, sum_ellipann, _sum_circle, _sum_ellipse


__all__ = ['aperture_photometry']


@njit(nogil=True, error_model='numpy', cache=True)
def calc_flux_err(a: np.ndarray, img_back: np.ndarray, x: float, y: float, elongation: float, theta: float,
                  err: np.ndarray, mask: np.ndarray, gain: float) -> float:
    """
    Calculate flux error for a source depending on the aperture size; used by adaptive photometry to calculate
    the optimal aperture

    :param a: aperture size as 1D 1-element array
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
    aper = np.empty(4, np.float64)
    aper[0] = a[0]
    aper[1] = a[0]/elongation
    aper[2] = theta
    aper[3] = 1
    flux, flux_err, _, flags = _sum_ellipse(x, y, aper, img_back, mask, 0, err, None, 0, gain)
    if flags or flux <= 0:
        raise ValueError()
    # noinspection PyTypeChecker
    return flux_err/flux


@njit(nogil=True, cache=True, parallel=True)
def isophotal_analysis(img_back: np.ndarray, x: np.ndarray, y: np.ndarray, radius: float):
    """
    Perform isophotal analysis at the given (X,Y) positions

    :param img_back: original image with background subtracted
    :param x: 1D array of X coordinates (0-based) of sources
    :param y: 1D array of Y coordinates (0-based) of sources, same shape as `x`
    :param radius: analysis radius in pixels
    :return:
        - array of semi-major axes in pixels
        - array of semi-minor axes in pixels
        - array of pisition angles in radians
        - array of fluxes
    """
    h, w = img_back.shape
    n = len(x)
    a = np.empty(n, np.float64)
    b = np.empty(n, np.float64)
    theta = np.empty(n, np.float64)
    flux = np.empty(n, np.float64)

    # np.indices() for 2D array
    xx = np.empty_like(img_back, np.int64)
    yy = np.empty_like(img_back, np.int64)
    for i in prange(w):
        xx[:, i] = i
    for i in prange(h):
        yy[i] = i

    for k in prange(n):
        ap = (xx - x[k])**2 + (yy - y[k])**2 <= radius**2
        if ap.any():
            yi, xi = ap.nonzero()
            nap = len(xi)
            ap_data = np.empty(nap, np.float64)
            ofs = 0
            for i in range(h):
                for j in range(w):
                    if ap[i, j]:
                        ap_data[ofs] = img_back[i, j]
                        ofs += 1
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
                thetai = np.arctan(2*xy/(x2 - y2))/2
                if y2 > x2:
                    thetai += np.pi/2
            m1 = (x2 + y2)/2
            m2 = np.sqrt(max((x2 - y2)**2/4 + xy**2, 0))
            ai = max(1/12, np.sqrt(max(m1 + m2, 0)))
            bi = max(1/12, np.sqrt(max(m1 - m2, 0)))
            if ai/bi > 3:
                # Prevent too elongated apertures usually occurring for faint objects
                bi = ai
            if ai < bi:
                ai, bi = bi, ai
                thetai += np.pi/2
            thetai %= np.pi
            if thetai > np.pi/2:
                thetai -= np.pi
        else:
            # Cannot obtain a,b,theta from isophotal analysis, assume circular aperture
            ai, bi, thetai, f = radius, radius, 0, 0

        a[k] = ai
        b[k] = bi
        theta[k] = thetai
        flux[k] = f

    return a, b, theta, flux


def aperture_photometry(img: np.ndarray | np.ma.MaskedArray,
                        sources: np.ndarray,
                        background: np.ndarray | np.ma.MaskedArray | None = None,
                        background_rms: np.ndarray | np.ma.MaskedArray | None = None,
                        texp: float = 1,
                        gain: float | np.ndarray | np.ma.MaskedArray = 1,
                        sat_level: float = 63000,
                        a: float | None = None,
                        b: float | None = None,
                        theta: float | None = 0,
                        a_in: float | None = None,
                        a_out: float | None = None,
                        b_out: float | None = None,
                        theta_out: float | None = None,
                        k: float = 0,
                        k_in: float | None = None,
                        k_out: float | None = None,
                        radius: float = 6,
                        fix_aper: bool = False,
                        fix_ell: bool = True,
                        fix_rot: bool = True,
                        apcorr_tol: float = 0.0001,
                        reject_outliers: bool = False) -> np.ndarray:
    """
    Do automatic or fixed aperture photometry

    :param img: input 2D image array
    :param sources: record array of sources extracted with :func:`skylib.extraction.extract_sources`; should contain
        at least "x" and "y" columns
    :param background: optional sky background map; if omitted, extract background from the annulus around the aperture,
        see `a_in` below
    :param background_rms: optional sky background RMS map; if omitted, calculate RMS over the annulus around
        the aperture
    :param texp: exposure time in seconds
    :param gain: electrons to data units conversion factor; used to estimate photometric errors; for variable-gain
        images (e.g. mosaics), must be an array of the same shape as the input data
    :param sat_level: saturation level in ADUs; used to select only non-saturated stars for adaptive aperture photometry
        and aperture correction
    :param a: fixed aperture radius or semi-major axis in pixels; default: use automatic photometry with a = a_iso*k,
        where a_iso is the isophotal semi-major axis
    :param b: semi-minor axis in pixels when using a fixed aperture; default: same as `a`
    :param theta: rotation angle of semi-major axis in degrees CCW when using a fixed aperture and `b` != `a`;
        default: 0
    :param a_in: inner annulus radius or semi-major axis in pixels; used to estimate the background if `background` or
        `background_rms` are not provided, ignored otherwise; default: `a`*`k_in`
    :param a_out: outer annulus radius or semi-major axis in pixels; default: `a`*`k_out`
    :param b_out: outer annulus semi-minor axis in pixels; default: `b`*`k_out`
    :param theta_out: annulus orientation in degrees CCW; default: same as `theta`
    :param k: automatic aperture radius in units of isophotal radius; 0 means find the optimal radius based on SNR;
        default: 0
    :param k_in: inner annulus radius in units of aperture radius (fixed aperture, i.e. `a` is provided) or isophotal
        radius (adaptive aperture); default: 1.5*`k` or 3.75 if `k` is undefined and `a` = None
    :param k_out: outer annulus radius in units of aperture radius (fixed aperture) or isophotal radius (adaptive
        aperture); default: 2*`k` or 5 if `k` is undefined and `a` = None
    :param radius: isophotal analysis radius in pixels used to compute automatic aperture if ellipse parameters
        (a,b,theta) are missing
    :param fix_aper: use the same aperture radius for all sources when doing automatic photometry; calculated as
        flux-weighted median of aperture sizes based on isophotal parameters
    :param fix_ell: use the same major to minor aperture axis ratio for all sources during automatic photometry;
        calculated as flux-weighted median of all ellipticities
    :param fix_rot: use the same aperture position angle for all sources during automatic photometry; calculated as
        flux-weighted median of all orientations
    :param apcorr_tol: growth curve stopping tolerance for aperture correction; 0 = disable aperture correction
    :param reject_outliers: if using annulus for background estimation, reject outlying pixels in the annulus; jgnored
        if `background` is supplied

    :return: record array containing the input sources, with the following fields added or updated: "flux", "flux_err",
        "mag", "mag_err", "aper_a", "aper_b", "aper_theta", "aper_a_in", "aper_a_out", "aper_b_out", "aper_theta_out",
        "aper_area", "background_area", "background", "background_rms", "phot_flag"
    """
    nsource = len(sources)
    if not nsource:
        return np.empty((0,), np.float64)

    img = sep_compatible(img)

    texp = float(texp)
    if np.isscalar(gain):
        gain = float(gain)
    k = float(k)
    if k <= 0.1:
        k = 0  # temporary fix for k = 0 not being allowed in AgA
    if k_in:
        k_in = float(k_in)
    elif k:
        k_in = 1.5*k
    else:
        k_in = 3.75
    if k_out:
        k_out = float(k_out)
    elif k:
        k_out = 2*k
    else:
        k_out = 5

    x, y = sources['x'] - 1, sources['y'] - 1
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
    z = np.zeros(nsource, np.float64)

    fixed_aper = bool(a)
    if fixed_aper:
        # Use the same fixed aperture and annulus parameters for all sources
        a = float(a)
        if b:
            b = float(b)
        else:
            b = a
        if theta:
            theta = float(theta) % 180
            if theta > 90:
                theta -= 180
            theta *= np.pi/180
        else:
            theta = 0

        if not have_background or apcorr_tol > 0:
            if theta_out:
                theta_out = float(theta_out) % 180
                if theta_out > 90:
                    theta_out -= 180
                theta_out *= np.pi/180
            elif theta_out != 0:
                theta_out = theta
            if a_in:
                a_in = float(a_in)
            else:
                a_in = a*k_in
            if a_out:
                a_out = float(a_out)
            else:
                a_out = a*k_out
            if b_out:
                b_out = float(b_out)
            else:
                b_out = a_out*b/a
    else:
        # Use automatic apertures derived from ellipse axes; will need image with background subtracted
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
            a[bad], b[bad], theta[bad], flux[bad] = isophotal_analysis(
                img_back, sources['x'][bad], sources['y'][bad], radius)
        else:
            theta %= np.pi
            theta[theta > np.pi/2] -= np.pi
        elongation = np.zeros_like(a)
        good = b > 0
        elongation[good] = a[good]/b[good]
        sources['a'] = a
        sources['b'] = b
        sources['theta'] = theta
        sources['flux'] = flux

        # Obtain the optimal aperture radius from the brightest non-saturated source
        if not k:
            aper = np.empty(4, np.float64)
            aper[3] = 1
            noise = np.zeros((1, 1), np.float64)
            for i in np.argsort(flux)[::-1]:
                aper[:3] = a[i], b[i], theta[i]
                if _sum_ellipse(x[i], y[i], aper, img >= sat_level, None, 0, noise, None, 0, 0)[0]:
                    # Saturated source
                    continue
                try:
                    # noinspection PyTypeChecker
                    res = minimize(
                        calc_flux_err, np.array([a[i]*1.6]),
                        (img_back, x[i], y[i], elongation[i], theta[i], tmp_rms, mask, gain),
                        bounds=[(1, None)], tol=1e-5)
                except ValueError:
                    continue
                if not res.success:
                    continue
                k = res.x[0]/a[i]
                break
            if not k:
                raise ValueError('Not enough data for automatic aperture factor; use explicit aperture factor')

        # k_in, k_out are in units of isophotal radius; convert to aperture radius units
        k_in /= k
        k_out /= k

        # Calculate weighted median of aperture sizes, elongations, and/or orientations if requested
        r = np.sqrt(a*b)*k
        if r.size > 1 and any([fix_aper, fix_ell, fix_rot]):
            flux[flux < 0] = 0
            if not flux.any():
                raise ValueError(
                    'Not enough data for weighted median in fixed-aperture automatic photometry; use static '
                    'fixed-aperture or fully adaptive automatic photometry instead')
            if fix_aper:
                r = weighted_median(r, flux)
            if fix_ell:
                elongation = weighted_median(elongation, flux)
            if fix_rot:
                theta = weighted_median(theta, flux, period=np.pi) % np.pi
                if theta > np.pi/2:
                    theta -= np.pi

        # Calculate the final aperture and annulus sizes
        sqrt_el = np.sqrt(elongation)
        a, b = r*sqrt_el, r/sqrt_el
        if not have_background or apcorr_tol > 0:
            a_in = a*k_in
            a_out, b_out = a*k_out, b*k_out
            theta_out = theta

    # Calculate mean and RMS of background; to get the pure sigma, set error to 1 and don't pass the gain; use
    # sum_*() instead of analytic expressions to calculate the aperture and annulus area accounting for masked
    # pixels and edges
    if have_background:
        # No outlier rejection needed since the background map is expected to be smooth
        if fixed_aper and a == b:
            bk_flux, _, bk_area, _ = sum_circle(background, x, y, a, err=1, mask=mask)
        else:
            bk_flux, _, bk_area, _ = sum_ellipse(background, x, y, a, b, theta, 1, err=1, mask=mask)
    elif fixed_aper and a_out == b_out:
        bk_flux, _, bk_area, _ = sum_circann(img, x, y, a_in, a_out, mask=mask, reject_outliers=reject_outliers)
    else:
        bk_flux, _, bk_area, _ = sum_ellipann(
            img, x, y, a_in, a_in*b_out/a_out, theta_out, 1, a_out/a_in, mask=mask, reject_outliers=reject_outliers)

    if fixed_aper and a == b:
        # Fixed circular aperture
        if have_background:
            flux, flux_err, area, flags = sum_circle(img, x, y, a, err=background_rms, mask=mask, gain=gain)
        else:
            flux, flux_err, area, flags = sum_circle(
                img, x, y, a, mask=mask, bkgann=(a_in, a_out), gain=gain, reject_outliers=reject_outliers)
    else:
        # Variable or elliptic aperture
        if have_background:
            flux, flux_err, area, flags = sum_ellipse(
                img, x, y, a, b, theta, 1, err=background_rms, mask=mask, gain=gain)
        else:
            flux, flux_err, area, flags = sum_ellipse(
                img, x, y, a, b, theta, 1, mask=mask, bkgann=(a_in/a, a_out/a), gain=gain,
                reject_outliers=reject_outliers)

    if have_background:
        # Subtract background from fluxes; background area equals aperture area
        flux -= bk_flux
        bk_mean = bk_flux/area
    else:
        # Background already subtracted; background area equals annulus area
        bk_mean = bk_flux/bk_area

    # Convert ADUs to electrons
    flux *= gain
    flux_err *= gain
    bk_mean *= gain

    # Calculate aperture correction for all aperture sizes from the brightest source
    aper_corr = {}
    if apcorr_tol > 0:
        aper_circle = np.empty(1, np.float64)
        aper_ellipse = np.empty(4, np.float64)
        aper_ellipse[3] = 1
        noise = np.zeros((1, 1), np.float64)
        for i in np.argsort(flux)[::-1]:
            xi, yi = x[i], y[i]
            if np.isscalar(a):
                ai = a
            else:
                ai = a[i]
            if np.isscalar(b):
                bi = b
            else:
                bi = b[i]
            if np.isscalar(theta):
                thetai = theta
            else:
                thetai = theta[i]
            if np.isscalar(a_in):
                ai_in = a_in
            else:
                ai_in = a_in[i]
            if np.isscalar(a_out):
                ai_out = a_out
            else:
                ai_out = a_out[i]

            if ai == bi:
                aper_circle[0] = ai
                nsat = _sum_circle(xi, yi, aper_circle, img >= sat_level, None, 0, noise, None, 0, 0)[0]
            else:
                aper_ellipse[:3] = ai, bi, thetai
                nsat = _sum_ellipse(xi, yi, aper_ellipse, img >= sat_level, None, 0, noise, None, 0, 0)[0]
            if nsat:
                # Saturated source
                continue

            # Obtain total flux by increasing aperture size until it grows either more than before (i.e. a nearby source
            # in the aperture) or less than the threshold (i.e. the growth curve reached saturation)
            if ai == bi:
                f, _, _, fl = sum_circle(
                    img, xi, yi, ai, mask=mask, bkgann=(ai_in, a_out), reject_outliers=reject_outliers)
            else:
                f, _, _, fl = sum_ellipse(
                    img, xi, yi, ai, bi, thetai, 1, mask=mask, bkgann=(ai_in, ai_out), reject_outliers=reject_outliers)
            if fl[0]:
                continue
            f0 = f_prev = f[0]
            dap = 0
            f_tot = None
            while dap < 100*ai:
                dap += 0.1
                if ai == bi:
                    f, _, _, fl = sum_circle(
                        img, xi, yi, ai + dap, mask=mask, bkgann=(ai_in + dap, ai_out + dap),
                        reject_outliers=reject_outliers)
                else:
                    f, _, _, fl = sum_ellipse(
                        img, xi, yi, ai + dap, bi*(1 + dap/ai), thetai, 1, mask=mask,
                        bkgann=(ai_in + dap, ai_out + dap), reject_outliers=reject_outliers)
                if fl[0]:
                    break
                f = f[0]
                if f <= 0:
                    break
                df = f/f_prev
                if df < 1:
                    # Decreasing flux; ignore this point
                    f_prev = f
                    continue
                if df < 1 + apcorr_tol:
                    # Growth stopped to within the tolerance
                    f_tot = f
                    break
                f_prev, df_prev = f, df
            if f_tot is None:
                continue

            # Calculate fluxes for the chosen source for all unique aperture sizes used for other sources
            fluxes_for_ap = {ai: f0}
            if not np.isscalar(a):
                for aj in set(a) - {ai}:
                    if ai == bi:
                        f, _, _, fl = sum_circle(
                            img, xi, yi, aj, mask=mask, bkgann=(aj*k_in, aj*k_out), reject_outliers=reject_outliers)
                    else:
                        f, _, _, fl = sum_ellipse(
                            img, xi, yi, aj, aj*bi/ai, thetai, 1, mask=mask, bkgann=(aj*k_in, aj*k_out),
                            reject_outliers=reject_outliers)
                    if fl[0] or f[0] <= 0:
                        continue
                    fluxes_for_ap[aj] = f[0]

            # Calculate aperture corrections
            for aj, f in fluxes_for_ap.items():
                if f < f_tot:
                    aper_corr[aj] = -2.5*np.log10(f_tot/f)

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

    for name in ['mag', 'mag_err', 'aper_a', 'aper_b', 'aper_theta', 'aper_a_in', 'aper_a_out', 'aper_b_out',
                 'aper_theta_out', 'aper_area', 'background_area', 'background', 'background_rms']:
        if name not in sources.dtype.names:
            sources = append_fields(sources, name, z, usemask=False)

    good = (flux > 0).nonzero()
    if len(good[0]):
        sources['mag'][good] = -2.5*np.log10(flux[good]/texp)
        sources['mag_err'][good] = 2.5*np.log10(1 + flux_err[good]/flux[good])

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
