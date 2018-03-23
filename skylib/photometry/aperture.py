"""
High-level aperture photometry interface.

:func:`~aperture_photometry()`: fixed or automatic (Kron-like) aperture
photometry of an image after source extraction.
"""

from __future__ import absolute_import, division, print_function

from numpy import clip, int32, isscalar, log10, ones, pi, zeros
from numpy.lib.recfunctions import append_fields
import sep
from ..calibration.background import sep_compatible


__all__ = ['aperture_photometry']


def aperture_photometry(img, sources, background=None, background_rms=None,
                        texp=1, gain=1, a=None, b=None, theta=0, a_in=None,
                        a_out=None, b_out=None, theta_out=None, k=2.5,
                        k_in=None, k_out=None):
    """
    Do automatic (Kron-like) or fixed aperture photometry

    :param array_like img: input 2D image array
    :param array_like sources: record array of sources extracted with
        :func:`skylib.extraction.extract_sources`; the function adds the
        following columns: flux, flux_err, mag, mag_err, aper_a, aper_b,
        aper_theta, aper_a_in, aper_a_out, aper_b_out, aper_theta_out,
        aper_area, background_area, background, background_rms
    :param array_like background: optional sky background map; if omitted,
        extract background from the annulus around the aperture, see `a_in`
        below
    :param array_like background_rms: optional sky background RMS map; if
        omitted, calculate RMS over the annulus around the aperture
    :param float texp: exposure time in seconds
    :param array_like gain: electrons to data units conversion factor; used to
        estimate photometric errors; for variable-gain images (e.g. mosaics),
        must be an array of the same shape as the input data
    :param float a: fixed aperture radius or semi-major axis in pixels;
        default: use automatic photometry with a = a_iso*k, where a_iso is the
        isophotal semi-major axis sigma
    :param float b: semi-minor axis in pixels when using a fixed aperture;
        default: same as `a`
    :param float theta: rotation angle of semi-major axis in degrees CCW when
        using a fixed aperture and `b` != `a`; default: 0
    :param float a_in: inner annulus radius or semi-major axis in pixels; used
        to estimate the background if `background` or `background_rms` are not
        provided, ignored otherwise; default: `a`*`k_in`
    :param float a_out: outer annulus radius or semi-major axis in pixels;
        default: `a`*`k_out`
    :param float b_out: outer annulus semi-minor axis in pixels;
        default: `b`*`k_out`
    :param float theta_out: annulus orientation in degrees CCW; default: same
        as `theta`
    :param float k: automatic aperture radius in units of isophotal radius;
        default: 2.5
    :param float k_in: inner annulus radius in units of isophotal radius;
        default: 1.5*`k`
    :param float k_out: outer annulus radius in units of isophotal radius;
        default: 2*`k`

    :return: record array containing the input sources, with "flux" and
        "flux_err" updated and the following fields added: "mag", "mag_err",
        "aper_a", "aper_b", "aper_theta", "aper_a_in", "aper_a_out",
        "aper_b_out", "aper_theta_out", "aper_area", "background_area",
        "background", "background_rms", "phot_flag"
    :rtype: numpy.ndarray
    """
    if not len(sources):
        return

    img = sep_compatible(img)

    texp = float(texp)
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

    have_background = background is not None and background_rms is not None
    if have_background:
        background = sep_compatible(background)
        background_rms = sep_compatible(background_rms)

    fixed_aper = bool(a)
    if fixed_aper:
        # Use the same fixed aperture and annulus parameters for all sources
        a = float(a)
        if b:
            b = float(b)
        else:
            b = a
        if theta:
            theta = float(theta)*pi/180
        else:
            theta = 0

        if not have_background:
            if theta_out:
                theta_out = float(theta_out)*pi/180
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
        theta = sources['theta']
        kron_r = clip(sep.kron_radius(
            img, x, y, sources['a'], sources['b'], theta, 6.0)[0], 0.1, None)
        elongation = sources['a']/sources['b']
        r = kron_r*k
        a, b = r*elongation, r/elongation
        if not have_background:
            a_in = a*(k_in/k)
            a_out, b_out = a*(k_out/k), b*(k_out/k)
            theta_out = theta

    if have_background:
        if fixed_aper and a == b:
            bk_area = sep.sum_circle(area_img, x, y, a, subpix=0)[0]
            bk_mean, bk_sigma = sep.sum_circle(
                background, x, y, a, subpix=0)[:2]
        else:
            # noinspection PyArgumentList
            bk_area = sep.sum_ellipse(area_img, x, y, a, b, theta, subpix=0)[0]
            # noinspection PyArgumentList
            bk_mean, bk_sigma = sep.sum_ellipse(
                background, x, y, a, b, theta, subpix=0)[:2]
        error = background_rms
    elif fixed_aper and a_out == b_out:
        bk_area = sep.sum_circann(area_img, x, y, a_in, a_out, subpix=0)[0]
        bk_mean, bk_sigma = sep.sum_circann(
            img, x, y, a_in, a_out, subpix=0)[:2]
        error = bk_sigma
    else:
        bk_area = sep.sum_ellipann(
            area_img, x, y, a_out, b_out, theta_out, a_in/a_out, 1, subpix=0)[0]
        bk_mean, bk_sigma = sep.sum_ellipann(
            img, x, y, a_out, b_out, theta_out, a_in/a_out, 1, subpix=0)[:2]
        error = bk_sigma

    if have_background:
        area = bk_area
    elif fixed_aper and a == b:
        area = sep.sum_circle(area_img, x, y, a, subpix=0)[0]
    else:
        # noinspection PyArgumentList
        area = sep.sum_ellipse(area_img, x, y, a, b, theta, subpix=0)[0]

    if not have_background:
        # We have a single error value per source; compute flux separately
        # for each source since SEP does not accept a 1D array of errors
        n = len(sources)
        flux, flux_err = zeros([2, n], dtype=float)
        flags = zeros(n, dtype=int)
    else:
        # We have a 2D error array; compute all fluxes at once
        flux = flux_err = flags = None

    if fixed_aper and a == b:
        # Fixed circular aperture
        if have_background:
            flux, flux_err, flags = sep.sum_circle(
                img, x, y, a, err=error, gain=gain, subpix=0)
        else:
            for i, (_x, _y, _err) in enumerate(zip(x, y, error)):
                flux[i], flux_err[i], flags[i] = sep.sum_circle(
                    img, _x, _y, a, err=_err, gain=gain, subpix=0)
    else:
        # Variable or elliptic aperture
        if have_background:
            # noinspection PyArgumentList
            flux, flux_err, flags = sep.sum_ellipse(
                img, x, y, a, b, theta, err=error, gain=gain, subpix=0)
        else:
            if all(isscalar(item) for item in (a, b, theta)):
                # Same aperture for all sources
                for i, (_x, _y, _err) in enumerate(zip(x, y, error)):
                    # noinspection PyArgumentList
                    flux[i], flux_err[i], flags[i] = sep.sum_ellipse(
                        img, _x, _y, a, b, theta, err=_err, gain=gain, subpix=0)
            else:
                # Individual aperture for each source
                if isscalar(a):
                    a = zeros(len(sources)) + a
                if isscalar(b):
                    b = zeros(len(sources)) + b
                if isscalar(theta):
                    theta = zeros(len(sources)) + theta
                for i, (_x, _y, _a, _b, _theta, _err) in enumerate(
                        zip(x, y, a, b, theta, error)):
                    # noinspection PyArgumentList
                    flux[i], flux_err[i], flags[i] = sep.sum_ellipse(
                        img, _x, _y, _a, _b, _theta, err=_err, gain=gain,
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

    sources['flux'] = flux
    sources['flag'] |= flags

    sources = append_fields(
        sources,
        ['flux_err', 'mag', 'mag_err', 'aper_a', 'aper_b', 'aper_theta',
         'aper_a_in', 'aper_a_out', 'aper_b_out', 'aper_theta_out', 'aper_area',
         'background_area', 'background', 'background_rms'],
        [flux_err] + [zeros(len(sources), float)]*13, usemask=False)

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
