"""
Source centroiding.

:func:`~centroid_sources()`: given the initial guess, obtain a more accurate
source centroid position using either SExtractor or IRAF method.
"""

from __future__ import absolute_import, division, print_function

import numpy
import sep

from ..calibration.background import sep_compatible


__all__ = ['centroid_iraf', 'centroid_sources']


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
            marg = box.sum(axis, dtype=float)/box.shape[axis]
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
        if abs(xc - xc_old) < tol and abs(yc - yc_old) < tol:
            break

    return xc + 1, yc + 1


def centroid_sources(data, x, y, radius=5, method='iraf'):
    """
    Given the initial guess, obtain a more accurate source centroid position(s)
    using either SExtractor or IRAF method

    :param array_like data: 2D pixel data array
    :param array_like x: initial guess for the source X position (1-based)
    :param array_like y: initial guess for the source Y position (1-based)
    :param array_like radius: centroiding radius, either an array of the same
        shape as `x` and `y` or a scalar if using the same radius for all
        sources
    :param str method: "iraf" (default) or "win" (windowed method, SExtractor)

    :return: (x, y) - a pair of centroid coordinates, same shape as input
    """
    if method == 'win':
        data = sep_compatible(data)
        xc, yc, flags = sep.winpos(data, x - 1, y - 1, radius)
        if numpy.ndim(flags):
            bad = flags.nonzero()
            xc[bad] = x[bad] - 1
            yc[bad] = y[bad] - 1
            return xc + 1, yc + 1
        if flags:
            return x, y
        return xc + 1, yc + 1

    if method == 'iraf':
        x, y = zip(*[
            centroid_iraf(data, x0, y0, r)
            for x0, y0, r in numpy.transpose(
                [numpy.atleast_1d(x), numpy.atleast_1d(y),
                 numpy.full_like(numpy.atleast_1d(x), radius)])])
        if not numpy.ndim(x):
            x, y = x[0], y[0]
        return x, y

    raise ValueError('Unknown centroiding method "{}"'.format(method))
