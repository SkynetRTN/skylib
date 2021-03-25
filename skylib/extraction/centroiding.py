"""
Source centroiding.

:func:`~centroid_iraf()`: given the initial guess, obtain a more accurate
source centroid position using the IRAF-like method.

:func:`~centroid_psf()`: given the initial guess, obtain a more accurate source
centroid position using Gaussian PSF fitting

:func:`~centroid_sources()`: given the initial guess, obtain a more accurate
source centroid position using SExtractor, IRAF, or PSF fitting method.
"""

from typing import Tuple, Union

import numpy
import sep
from scipy.optimize import leastsq
from astropy.stats import gaussian_fwhm_to_sigma

from ..calibration.background import sep_compatible


__all__ = ['centroid_iraf', 'centroid_psf', 'centroid_sources']


def centroid_iraf(data: Union[numpy.ndarray, numpy.ma.MaskedArray],
                  x: float, y: float, radius: float = 5, tol: float = 0.2,
                  max_iter: int = 10) -> Tuple[float, float]:
    """
    Given the initial guess, obtain a more accurate source centroid position
    using the IRAF-like method

    :param data: 2D pixel data array
    :param x: initial guess for the source X position (1-based)
    :param y: initial guess for the source Y position (1-based)
    :param radius: centroiding radius
    :param tol: position tolerance; stop if both X and Y centroid coordinates
        change by less than this value with respect to the previous iteration
    :param int max_iter: maximum number of iterations

    :return: (x, y) - a pair of centroid coordinates
    """
    h, w = data.shape
    xc, yc = x - 1, y - 1
    for _ in range(max_iter):
        x1 = min(max(int(xc - radius + 0.5), 0), w - 1)
        y1 = min(max(int(yc - radius + 0.5), 0), h - 1)
        x2 = min(max(int(xc + radius + 0.5), 0), w - 1)
        y2 = min(max(int(yc + radius + 0.5), 0), h - 1)
        if x1 > x2 or y1 > y2:
            break
        box = data[y1:y2 + 1, x1:x2 + 1]
        box = box - box.min()

        xy = []
        for axis in (0, 1):
            marg = box.mean(axis)
            marg -= marg.mean()
            good = (marg > 0).nonzero()
            if not len(good[0]):
                break
            marg = marg[good]
            xy.append(numpy.dot(
                numpy.arange((x1, y1)[axis] + 1, (x2, y2)[axis] + 2)[good],
                marg)/marg.sum() - 1)
        if len(xy) < 2 or xy[0] < 0 or xy[0] >= w or xy[1] < 0 or xy[1] >= h:
            break

        xc_old, yc_old = xc, yc
        xc, yc = xy
        if max(abs(xc - xc_old), abs(yc - yc_old)) < tol:
            break

    return float(xc) + 1, float(yc) + 1


def gauss_ellip(x: numpy.ndarray, y: numpy.ndarray, p: numpy.ndarray) \
        -> numpy.ndarray:
    """
    Elliptical Gaussian PSF

    :param x: array of X coordinates
    :param y: array of Y coordinates
    :param p: 7-element array of parameters: (x0, y0, baseline, amplitude,
        sigma_x, sigma_y, theta)
    :return:
    """
    x0, y0, baseline, ampl, s1, s2, theta = p
    sn, cs = numpy.sin(theta), numpy.cos(theta)
    a = cs**2/s1 + sn**2/s2
    b = sn**2/s1 + cs**2/s2
    c = 2*sn*cs*(1/s1 - 1/s2)
    dx, dy = x - x0, y - y0
    return baseline + ampl*numpy.exp(-0.5*(a*dx**2 + b*dy**2 + c*dx*dy))


def centroid_psf(data: Union[numpy.ndarray, numpy.ma.MaskedArray],
                 x: float, y: float, radius: float = 5, ftol: float = 1e-4,
                 xtol: float = 1e-4, maxfev: int = 1000) -> Tuple[float, float]:
    """
    Given the initial guess, obtain a more accurate source centroid position
    and ellipse parameters using Gaussian PSF fitting

    :param data: 2D pixel data array
    :param x: initial guess for the source X position (1-based)
    :param y: initial guess for the source Y position (1-based)
    :param radius: centroiding radius
    :param ftol: relative error desired in the sum of squares (see
        :func:`scipy.optimize.leastsq`)
    :param xtol: relative error desired in the approximate solution
    :param maxfev: maximum number of calls to the function

    :return: (x, y) - a pair of centroid coordinates, same shape as input
    """
    h, w = data.shape
    xc, yc = x - 1, y - 1
    radius = max(radius, 3)
    x1 = min(max(int(xc - radius + 0.5), 0), w - 1)
    y1 = min(max(int(yc - radius + 0.5), 0), h - 1)
    x2 = min(max(int(xc + radius + 0.5), 0), w - 1)
    y2 = min(max(int(yc + radius + 0.5), 0), h - 1)
    box = data[y1:y2 + 1, x1:x2 + 1]

    # Keep only data within the circle centered at (xc,yc)
    x0, y0 = xc - x1, yc - y1
    y, x = numpy.indices(box.shape)
    circ = (x - x0)**2 + (y - y0)**2 <= radius**2
    box = box[circ].ravel().copy()
    if len(box) < 8:
        # Not enough pixels within the aperture to get an overdetermined system
        # for all 7 PSF parameters
        return xc + 1, yc + 1
    box -= box.min()
    x, y = x[circ].ravel(), y[circ].ravel()

    # Initial guess
    ampl = box.max()
    sigma2 = (box > ampl/2).sum()*gaussian_fwhm_to_sigma**2

    # Get centroid position by least-squares fitting
    p = leastsq(
        lambda _p: gauss_ellip(x, y, _p) - box,
        numpy.array([xc - x1, yc - y1, 0, ampl, sigma2, sigma2, 0]),
        ftol=ftol, xtol=xtol, maxfev=maxfev)[0]

    return float(p[0]) + x1 + 1, float(p[1] + y1 + 1)


def centroid_sources(data: Union[numpy.ndarray, numpy.ma.MaskedArray],
                     x: Union[float, numpy.ndarray],
                     y: Union[float, numpy.ndarray],
                     radius: Union[float, numpy.ndarray] = 5,
                     method: str = 'iraf') \
        -> Union[Tuple[float, float], Tuple[numpy.ndarray, numpy.ndarray]]:
    """
    Given the initial guess, obtain a more accurate source centroid position(s)
    using SExtractor, IRAF, or PSF fitting method

    :param data: 2D pixel data array
    :param x: initial guess for the source X position (1-based)
    :param y: initial guess for the source Y position (1-based)
    :param radius: centroiding radius, either an array of the same shape as `x`
        and `y` or a scalar if using the same radius for all sources
    :param method: "iraf" (default), "win" (windowed method, SExtractor),
        or "psf" (Gaussian PSF fitting)

    :return: (x, y) - a pair of centroid coordinates, same shape as input
    """
    if method == 'win':
        data = sep_compatible(data)
        if isinstance(data, numpy.ma.MaskedArray):
            mask = data.mask
            data = data.data
        else:
            mask = None
        xc, yc, flags = sep.winpos(data, x - 1, y - 1, radius, mask=mask)
        if numpy.ndim(flags):
            bad = flags.nonzero()
            xc[bad] = x[bad] - 1
            yc[bad] = y[bad] - 1
            return xc + 1, yc + 1
        if flags:
            return x, y
        return xc + 1, yc + 1

    x, y = tuple(zip(*[
        (centroid_psf if method == 'psf' else centroid_iraf)(data, x0, y0, r)
        for x0, y0, r in numpy.transpose(
            [numpy.atleast_1d(x), numpy.atleast_1d(y),
             numpy.full_like(numpy.atleast_1d(x), radius)])]))
    if not numpy.ndim(x):
        x, y = x[0], y[0]
    return x, y
