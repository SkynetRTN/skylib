"""
High-level aperture photometry interface.

:func:`~aperture_photometry()`: fixed or automatic (Kron-like) aperture
photometry of an image after source extraction.
"""

from typing import Optional, Union

from numpy import (
    arctan, array, clip, empty, full_like, indices, int32, isscalar, log10,
    ndarray, ndim, ones, pi, sqrt, zeros)
from numpy.lib.recfunctions import append_fields
from numpy.ma import MaskedArray
import sep

from ..calibration.background import sep_compatible
from ..util.stats import weighted_median


__all__ = ['aperture_photometry']


def aperture_photometry(img: Union[ndarray, MaskedArray], sources: ndarray,
                        background: Optional[Union[ndarray,
                                                   MaskedArray]] = None,
                        background_rms: Optional[Union[ndarray,
                                                       MaskedArray]] = None,
                        texp: float = 1,
                        gain: Union[float, ndarray, MaskedArray] = 1,
                        a: Optional[float] = None, b: Optional[float] = None,
                        theta: Optional[float] = 0,
                        a_in: Optional[float] = None,
                        a_out: Optional[float] = None,
                        b_out: Optional[float] = None,
                        theta_out: Optional[float] = None,
                        k: float = 2.5,
                        k_in: Optional[float] = None,
                        k_out: Optional[float] = None,
                        radius: float = 6,
                        fix_aper: bool = True,
                        fix_ell: bool = True,
                        fix_rot: bool = True) -> ndarray:
    """
    Do automatic (Kron-like) or fixed aperture photometry

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
    :param a: fixed aperture radius or semi-major axis in pixels; default: use
        automatic photometry with a = a_iso*k, where a_iso is the isophotal
        semi-major axis sigma
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
    :param k: automatic aperture radius in units of isophotal radius;
        default: 2.5
    :param k_in: inner annulus radius in units of isophotal radius;
        default: 1.5*`k`
    :param k_out: outer annulus radius in units of isophotal radius;
        default: 2*`k`
    :param radius: isophotal analysis radius in pixels used for Kron aperture
        if ellipse parameters (a,b,theta) are missing
    :param fix_aper: use the same aperture radius for all sources when doing
        automatic photometry; calculated as flux-weighted median of apertures
        based on Kron radius
    :param fix_ell: use the same major to minor aperture axis ratio for all
        sources during automatic photometry; calculated as flux-weighted median
        of all ellipticities
    :param fix_rot: use the same aperture position angle for all sources during
        automatic photometry; calculated as flux-weighted median
        of all orientations

    :return: record array containing the input sources, with the following
        fields added or updated: "flux", "flux_err", "mag", "mag_err", "aper_a",
        "aper_b", "aper_theta", "aper_a_in", "aper_a_out", "aper_b_out",
        "aper_theta_out", "aper_area", "background_area", "background",
        "background_rms", "phot_flag"
    """
    if not len(sources):
        return array([])

    img = sep_compatible(img)

    texp = float(texp)
    if isscalar(gain):
        gain = float(gain)
    k = float(k)
    if k_in:
        k_in = float(k_in)
    else:
        k_in = 1.5*k
    if k_out:
        k_out = float(k_out)
    else:
        k_out = 2*k

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
        # Use automatic apertures derived from kron radius and ellipse axes
        for name in ['a', 'b', 'theta']:
            if name not in sources.dtype.names:
                sources = append_fields(sources, name, z, usemask=False)
        a, b, theta = sources['a'], sources['b'], sources['theta']
        bad = (a <= 0) | (b <= 0)
        if bad.any():
            # Do isophotal analysis to compute ellipse parameters if missing
            yy, xx = indices(img.shape)
            for i in bad.nonzero()[0]:
                ap = (xx - sources[i]['x'])**2 + (yy - sources[i]['y'])**2 <= \
                    radius**2
                if ap.any():
                    yi, xi = ap.nonzero()
                    ap_data = img[ap].astype(float)
                    flux = ap_data.sum()
                    if flux > 0:
                        cx = (xi*ap_data).sum()/flux
                        cy = (yi*ap_data).sum()/flux
                        x2 = (xi**2*ap_data).sum()/flux - cx**2
                        y2 = (yi**2*ap_data).sum()/flux - cy**2
                        xy = (xi*yi*ap_data).sum()/flux - cx*cy
                    else:
                        cx, cy = xi.mean(), yi.mean()
                        x2 = (xi**2).mean() - cx**2
                        y2 = (yi**2).mean() - cy**2
                        xy = (xi*yi).mean() - cx*cy
                    if x2 == y2:
                        thetai = 0
                    else:
                        thetai = arctan(2*xy/(x2 - y2))/2
                        if x2 > y2:
                            thetai += pi/2
                    m1 = (x2 + y2)/2
                    m2 = sqrt(max((x2 - y2)**2/4 + xy**2, 0))
                    ai = max(1/12, sqrt(max(m1 + m2, 0)))
                    bi = max(1/12, sqrt(max(m1 - m2, 0)))
                else:
                    # Cannot obtain a,b,theta from isophotal analysis, assume
                    # circular aperture
                    ai, bi, thetai = radius, radius, 0
                a[i] = sources[i]['a'] = ai
                b[i] = sources[i]['b'] = bi
                theta[i] = sources[i]['theta'] = thetai
        bad = (a < b).nonzero()
        a[bad], b[bad] = b[bad], a[bad]
        theta[bad] += pi/2
        theta %= pi
        theta[theta > pi/2] -= pi
        kron_r = clip(
            sep.kron_radius(img, x, y, a, b, theta, 6.0, mask=mask)[0],
            0.1, None)
        r = kron_r*k
        elongation = a/b

        if r.size > 1 and any([fix_aper, fix_ell, fix_rot]):
            # Need fluxes to compute weighted median(s)
            if 'flux' in sources.dtype.names and (sources['flux'] > 0).any():
                flux = sources['flux']
            else:
                # Run a preliminary fixed-aperture photometry pass to compute
                # fluxes if missing from input data
                flux = aperture_photometry(
                    img, sources, background=background,
                    background_rms=background_rms, texp=texp, gain=gain,
                    a=radius, a_in=radius*k_in/k, a_out=radius*k_out/k)['flux']
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

        a, b = r*elongation, r/elongation
        if not have_background:
            a_in = a*(k_in/k)
            a_out, b_out = a*(k_out/k), b*(k_out/k)
            theta_out = theta

    if have_background:
        if fixed_aper and a == b:
            bk_area = sep.sum_circle(area_img, x, y, a, mask=mask, subpix=0)[0]
            bk_mean, bk_sigma = sep.sum_circle(
                background, x, y, a, mask=mask, subpix=0)[:2]
        else:
            bk_area = sep.sum_ellipse(
                area_img, x, y, a, b, theta, 1, mask=mask, subpix=0)[0]
            bk_mean, bk_sigma = sep.sum_ellipse(
                background, x, y, a, b, theta, 1, mask=mask, subpix=0)[:2]
        error = background_rms
    elif fixed_aper and a_out == b_out:
        bk_area = sep.sum_circann(
            area_img, x, y, a_in, a_out, mask=mask, subpix=0)[0]
        bk_mean, bk_sigma = sep.sum_circann(
            img, x, y, a_in, a_out, mask=mask, subpix=0)[:2]
        error = bk_sigma
    else:
        bk_area = sep.sum_ellipann(
            area_img, x, y, a_out, b_out, theta_out, a_in/a_out, 1, mask=mask,
            subpix=0)[0]
        bk_mean, bk_sigma = sep.sum_ellipann(
            img, x, y, a_out, b_out, theta_out, a_in/a_out, 1, mask=mask,
            subpix=0)[:2]
        error = bk_sigma

    if have_background:
        area = bk_area
    elif fixed_aper and a == b:
        area = sep.sum_circle(area_img, x, y, a, mask=mask, subpix=0)[0]
    else:
        area = sep.sum_ellipse(
            area_img, x, y, a, b, theta, 1, mask=mask, subpix=0)[0]

    if fixed_aper and a == b:
        # Fixed circular aperture
        if ndim(error) == 1:
            # Separate scalar error for each source
            flux, flux_err = empty([2, len(sources)], dtype=float)
            flags = empty(len(sources), dtype=int)
            for i, (_x, _y, _err) in enumerate(zip(x, y, error)):
                flux[i], flux_err[i], flags[i] = sep.sum_circle(
                    img, [_x], [_y], a, err=_err, mask=mask, gain=gain,
                    subpix=0)
        else:
            flux, flux_err, flags = sep.sum_circle(
                img, x, y, a, err=error, mask=mask, gain=gain, subpix=0)
    else:
        # Variable or elliptic aperture
        if ndim(error) == 1:
            # Separate scalar error for each source
            flux, flux_err = empty([2, len(sources)], dtype=float)
            flags = empty(len(sources), dtype=int)
            if isscalar(a):
                a = full_like(x, a)
            if isscalar(b):
                b = full_like(x, b)
            if isscalar(theta):
                theta = full_like(x, theta)
            for i, (_x, _y, _err, _a, _b, _theta) in enumerate(zip(
                    x, y, error, a, b, theta)):
                flux[i], flux_err[i], flags[i] = sep.sum_ellipse(
                    img, [_x], [_y], _a, _b, _theta, 1, err=_err, mask=mask,
                    gain=gain, subpix=0)
        else:
            flux, flux_err, flags = sep.sum_ellipse(
                img, x, y, a, b, theta, 1, err=error, mask=mask, gain=gain,
                subpix=0)

    # Convert background sum to mean and subtract background from fluxes
    if have_background:
        # Background area equals aperture area
        flux -= bk_mean
        bk_mean = bk_mean/area
    else:
        # Background area equals annulus area
        bk_mean = bk_mean/bk_area
        flux -= bk_mean*area

    # Convert ADUs to electrons
    flux *= gain
    flux_err *= gain
    bk_mean *= gain
    bk_sigma *= gain

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
    sources['background_rms'] = bk_sigma

    return sources
