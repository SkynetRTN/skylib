"""
Source centroiding.

:func:`~centroid_iraf()`: given the initial guess, obtain a more accurate
source centroid position using the IRAF-like method.

:func:`~centroid_psf()`: given the initial guess, obtain a more accurate source
centroid position using Gaussian PSF fitting

:func:`~centroid_sources()`: given the initial guess, obtain a more accurate
source centroid position using SExtractor, IRAF, or PSF fitting method.
"""

from __future__ import absolute_import, division, print_function

import numpy
import sep
from scipy.optimize import leastsq

from ..calibration.background import sep_compatible


__all__ = ['centroid_iraf', 'centroid_psf', 'centroid_sources']


def centroid_iraf(data, x, y, radius=5, tol=0.2, max_iter=10):
    """
    Given the initial guess, obtain a more accurate source centroid position
    using the IRAF-like method

    :param array_like data: 2D pixel data array
    :param float x: initial guess for the source X position (1-based)
    :param float y: initial guess for the source Y position (1-based)
    :param float radius: centroiding radius
    :param float tol: position tolerance; stop if both X and Y centroid
        coordinates change by less than this value with respect to the previous
        iteration
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


def gauss_ellip(x, y, p):
    x0, y0, baseline, ampl, s1, s2, theta = p
    sn, cs = numpy.sin(theta), numpy.cos(theta)
    a = cs**2/s1 + sn**2/s2
    b = sn**2/s1 + cs**2/s2
    c = 2*sn*cs*(1/s1 - 1/s2)
    dx, dy = x - x0, y - y0
    return baseline + ampl*numpy.exp(-0.5*(a*dx**2 + b*dy**2 + c*dx*dy))


k_gauss = 2*numpy.sqrt(2*numpy.log(2))


def centroid_psf(data, x, y, radius=5, ftol=1e-4, xtol=1e-4, maxfev=1000):
    """
    Given the initial guess, obtain a more accurate source centroid position
    and ellipse parameters using Gaussian PSF fitting

    :param array_like data: 2D pixel data array
    :param float x: initial guess for the source X position (1-based)
    :param float y: initial guess for the source Y position (1-based)
    :param float radius: centroiding radius
    :param float ftol: relative error desired in the sum of squares (see
        :func:`scipy.optimize.leastsq`)
    :param float xtol: relative error desired in the approximate solution
    :param int maxfev: maximum number of calls to the function

    :return: (x, y, a, b, theta) - centroid coordinates plus ellipse semi-axes
        and position angle in degrees
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
    sigma2 = (box > ampl/2).sum()/k_gauss**2

    # Get centroid position by least-squares fitting
    p = leastsq(
        lambda _p: gauss_ellip(x, y, _p) - box,
        numpy.array([xc - x1, yc - y1, 0, ampl, sigma2, sigma2, 0]),
        ftol=ftol, xtol=xtol, maxfev=maxfev)[0]

    a, b = float(numpy.sqrt(p[4])), float(numpy.sqrt(p[5]))
    theta = float(numpy.rad2deg(p[6]))
    if a < b:
        # Make sure a is semi-major axis
        a, b = b, a
        theta += 90
    theta %= 180
    if theta > 90:
        theta -= 180

    return float(p[0]) + x1 + 1, float(p[1] + y1 + 1), a, b, theta


def centroid_sources(data, x, y, radius=5, method='iraf'):
    """
    Given the initial guess, obtain a more accurate source centroid position(s)
    using SExtractor, IRAF, or PSF fitting method

    :param array_like data: 2D pixel data array
    :param array_like x: initial guess for the source X position (1-based)
    :param array_like y: initial guess for the source Y position (1-based)
    :param array_like radius: centroiding radius, either an array of the same
        shape as `x` and `y` or a scalar if using the same radius for all
        sources
    :param str method: "iraf" (default), "win" (windowed method, SExtractor),
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
        (centroid_psf if method == 'psf' else centroid_iraf)(
            data, x0, y0, r,
        )[:2]
        for x0, y0, r in numpy.transpose(
            [numpy.atleast_1d(x), numpy.atleast_1d(y),
             numpy.full_like(numpy.atleast_1d(x), radius)])]))
    if not numpy.ndim(x):
        x, y = x[0], y[0]
    return x, y
