"""
Source centroiding.

:func:`~centroid_iraf()`: given the initial guess, obtain a more accurate
source centroid position using the IRAF-like method.

:func:`~centroid_psf()`: given the initial guess, obtain a more accurate source
centroid position using Gaussian PSF fitting

:func:`~centroid_sources()`: given the initial guess, obtain a more accurate
source centroid position using SExtractor, IRAF, or PSF fitting method.
"""

import sys

import numpy as np
import sep
from scipy.optimize import leastsq
from astropy.stats import gaussian_fwhm_to_sigma
from numba import njit, prange

from ..calibration.background import sep_compatible


__all__ = ['centroid_iraf', 'centroid_iraf_masked', 'centroid_psf', 'centroid_sources']


MAX_FLOAT = sys.float_info.max


@njit(nogil=True, parallel=True, cache=True)
def centroid_iraf(data: np.ndarray, x: np.ndarray, y: np.ndarray, radius: np.ndarray, tol: float = 0.2,
                  max_iter: int = 10) -> None:
    """
    Given the initial guess, obtain more accurate source centroid positions using the IRAF-like method

    :param data: 2D pixel data array; no NaNs/Infs allowed
    :param x: initial guess for the source X positions (1-based), 1D array; modified in place
    :param y: initial guess for the source Y positions (1-based), 1D array; modified in place
    :param radius: centroiding radii, same shape as `x` and `y`
    :param tol: position tolerance; stop if both X and Y centroid coordinates change by less than this value with
        respect to the previous iteration
    :param int max_iter: maximum number of iterations
    """
    h, w = data.shape
    for i in prange(len(x)):
        xc, yc = x[i] - 1, y[i] - 1
        r = radius[i]
        success = False
        for _ in range(max_iter):
            xc_old, yc_old = xc, yc
            x1 = min(max(int(np.floor(xc - r)), 0), w - 1)
            y1 = min(max(int(np.floor(yc - r)), 0), h - 1)
            x2 = min(max(int(np.ceil(xc + r)), 0), w - 1)
            y2 = min(max(int(np.ceil(yc + r)), 0), h - 1)
            if x1 > x2 or y1 > y2:
                break

            box = data[y1:y2 + 1, x1:x2 + 1] - data[y1:y2 + 1, x1:x2 + 1].min()
            bh, bw = box.shape

            marg = np.zeros(bw)
            for bx in range(bw):
                for by in range(bh):
                    marg[bx] += box[by, bx]
                marg[bx] /= bh
            marg -= marg.mean()
            good = marg > 0
            if not good.any():
                break
            marg = marg[good]
            xc = (np.arange(x1 + 1, x2 + 2)[good].astype(marg.dtype) @ marg)/marg.sum() - 1
            if xc < 0 or xc >= w:
                break

            marg = np.zeros(bh)
            for by in range(bh):
                for bx in range(bw):
                    marg[by] += box[by, bx]
                marg[by] /= bw
            marg -= marg.mean()
            good = marg > 0
            if not good.any():
                break
            marg = marg[good]
            yc = (np.arange(y1 + 1, y2 + 2)[good].astype(marg.dtype) @ marg)/marg.sum() - 1
            if yc < 0 or yc >= h:
                break

            if max(abs(xc - xc_old), abs(yc - yc_old)) < tol:
                success = True
                break

        if success:
            x[i] = xc + 1
            y[i] = yc + 1


@njit(nogil=True, parallel=True, cache=True)
def centroid_iraf_masked(data: np.ndarray, mask: np.ndarray, x: np.ndarray, y: np.ndarray, radius: np.ndarray,
                         tol: float = 0.2, max_iter: int = 10) -> None:
    """
    Same as `centroid_iraf` but works with masked arrays

    :param data: 2D pixel data array
    :param mask: 2D bool or uint8 array with masked element indicated by 1, same shape as `data`
    :param x: initial guess for the source X positions (1-based), 1D array; modified in place
    :param y: initial guess for the source Y positions (1-based), 1D array; modified in place
    :param radius: centroiding radii, same shape as `x` and `y`
    :param tol: position tolerance; stop if both X and Y centroid coordinates change by less than this value with
        respect to the previous iteration
    :param int max_iter: maximum number of iterations
    """
    h, w = data.shape
    for i in prange(len(x)):
        xc, yc = x[i] - 1, y[i] - 1
        r = radius[i]
        success = False
        for _ in range(max_iter):
            xc_old, yc_old = xc, yc
            x1 = min(max(int(np.floor(xc - r)), 0), w - 1)
            y1 = min(max(int(np.floor(yc - r)), 0), h - 1)
            x2 = min(max(int(np.ceil(xc + r)), 0), w - 1)
            y2 = min(max(int(np.ceil(yc + r)), 0), h - 1)
            if x1 > x2 or y1 > y2:
                break

            box = data[y1:y2 + 1, x1:x2 + 1].copy()
            bh, bw = box.shape
            box_mask = mask[y1:y2 + 1, x1:x2 + 1]
            min_val = MAX_FLOAT
            for by in range(bh):
                for bx in range(bw):
                    if not box_mask[by, bx]:
                        bv = box[by, bx]
                        if bv < min_val:
                            min_val = bv
            box -= min_val

            marg = np.zeros(bw)
            marg_good = np.ones(bw, np.bool_)
            for bx in range(bw):
                n = 0
                for by in range(bh):
                    if not box_mask[by, bx]:
                        marg[bx] += box[by, bx]
                        n += 1
                if n:
                    marg[bx] /= n
                else:
                    marg_good[bx] = False
            mean_val = 0.0
            n = 0
            for bx in range(bw):
                if marg_good[bx]:
                    mean_val += marg[bx]
                    n += 1
            if n:
                marg -= mean_val/n
            good = marg_good & (marg > 0)
            if not good.any():
                break
            marg = marg[good]
            xc = (np.arange(x1 + 1, x2 + 2)[good].astype(marg.dtype) @ marg)/marg.sum() - 1
            if xc < 0 or xc >= w:
                break

            marg = np.zeros(bh)
            marg_good = np.ones(bh, np.bool_)
            for by in range(bh):
                n = 0
                for bx in range(bw):
                    if not box_mask[by, bx]:
                        marg[by] += box[by, bx]
                        n += 1
                if n:
                    marg[by] /= n
                else:
                    marg_good[by] = False
            mean_val = 0.0
            n = 0
            for by in range(bh):
                if marg_good[by]:
                    mean_val += marg[by]
                    n += 1
            if n:
                marg -= mean_val/n
            good = marg_good & (marg > 0)
            if not good.any():
                break
            marg = marg[good]
            yc = (np.arange(y1 + 1, y2 + 2)[good].astype(marg.dtype) @ marg)/marg.sum() - 1
            if yc < 0 or yc >= h:
                break

            if max(abs(xc - xc_old), abs(yc - yc_old)) < tol:
                success = True
                break

        if success:
            x[i] = xc + 1
            y[i] = yc + 1


def gauss_ellip(x: np.ndarray, y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Elliptical Gaussian PSF

    :param x: array of X coordinates
    :param y: array of Y coordinates
    :param p: 7-element array of parameters: (x0, y0, baseline, amplitude, sigma_x^2, sigma_y^2, theta)

    :return: values of the function at the given points
    """
    x0, y0, baseline, ampl, s1, s2, theta = p
    sn, cs = np.sin(theta), np.cos(theta)
    a = cs**2/s1 + sn**2/s2
    b = sn**2/s1 + cs**2/s2
    c = 2*sn*cs*(1/s1 - 1/s2)
    dx, dy = x - x0, y - y0
    return baseline + ampl*np.exp(-0.5*(a*dx**2 + b*dy**2 + c*dx*dy))


def centroid_psf(data: np.ndarray | np.ma.MaskedArray, x: float, y: float, radius: float = 5, ftol: float = 1e-4,
                 xtol: float = 1e-4, maxfev: int = 1000) -> tuple[float, float]:
    """
    Given the initial guess, obtain a more accurate source centroid position and ellipse parameters using Gaussian PSF
    fitting

    :param data: 2D pixel data array
    :param x: initial guess for the source X position (1-based)
    :param y: initial guess for the source Y position (1-based)
    :param radius: centroiding radius
    :param ftol: relative error desired in the sum of squares (see :func:`scipy.optimize.leastsq`)
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
    y, x = np.indices(box.shape)
    circ = (x - x0)**2 + (y - y0)**2 <= radius**2
    box = box[circ].ravel().copy()
    if len(box) < 8:
        # Not enough pixels within the aperture to get an overdetermined system for all 7 PSF parameters
        return xc + 1, yc + 1
    box -= box.min()
    x, y = x[circ].ravel(), y[circ].ravel()

    # Initial guess
    ampl = box.max()
    sigma2 = (box > ampl/2).sum()*gaussian_fwhm_to_sigma**2

    # Get centroid position by least-squares fitting
    p = leastsq(
        lambda _p: gauss_ellip(x, y, _p) - box, np.array([xc - x1, yc - y1, 0, ampl, sigma2, sigma2, 0]),
        ftol=ftol, xtol=xtol, maxfev=maxfev)[0]

    return float(p[0]) + x1 + 1, float(p[1] + y1 + 1)


def centroid_sources(data: np.ndarray | np.ma.MaskedArray, x: np.ndarray, y: np.ndarray,
                     radius: float | np.ndarray = 5, method: str = 'iraf') -> None:
    """
    Given the initial guess, obtain more accurate source centroid positions using SExtractor, IRAF, or PSF fitting
    method

    :param data: 2D pixel data array
    :param x: initial guess for the source X positions (1-based); modified in place
    :param y: initial guess for the source Y positions (1-based), same shape as `x`; modified in place
    :param radius: centroiding radius, either an array of the same shape as `x` and `y` or a scalar if using the same
        radius for all sources
    :param method: "iraf" (default), "win" (windowed method, SExtractor), or "psf" (Gaussian PSF fitting)
    """
    x, y = np.ravel(x), np.ravel(y)
    radius = np.full_like(x, radius)

    if method == 'iraf':
        if isinstance(data, np.ma.MaskedArray) and data.mask is not False:
            centroid_iraf_masked(data.data, data.mask, x, y, radius)
        else:
            centroid_iraf(data, x, y, radius)

    elif method == 'win':
        data = sep_compatible(data)
        if isinstance(data, np.ma.MaskedArray):
            mask = data.mask
            data = data.data
        else:
            mask = None
        x1, y1, flags = sep.winpos(data, x - 1, y - 1, radius, mask=mask)
        good = flags == 0
        if good.all():
            x[:] = x1 + 1
            y[:] = y + 1
        else:
            x[good] = x1[good] + 1
            y[good] = y1[good] + 1

    elif method == 'psf':
        for i in range(len(x)):
            x[i], y[i] = centroid_psf(data, x[i], y[i], radius[i])

    else:
        raise ValueError(f'Unknown centroiding method "{method}"')
