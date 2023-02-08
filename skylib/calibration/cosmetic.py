"""
Numba-accelerated cosmetic defect correction

Adapted from the original code by Logan Selph.

:func:`correct_cosmetic`: high-level automatic intra-image correction pipeline.
:func:`flag_horiz`, :func:`flag_columns`, :func:`flag_pixels`: individual
    cosmetic defect rejection steps.
:func:`correct_cols_and_pixels`: eliminate cosmetic defects by local averaging
    using rejection maps obtained by :func:`flag_*`.
"""

import math

import numpy as np
from numba import njit, prange

from ..util.stats import chauvenet


__all__ = [
    'correct_cosmetic',
    'flag_horiz', 'flag_columns', 'flag_pixels', 'correct_cols_and_pixels',
]


@njit(nogil=True, parallel=True, cache=True)
def flag_horiz(img: np.ndarray, m: int = 10, nu: int = 0) -> np.ndarray:
    """
    Flag pixels with outlying values within their [-m,m] horizontal vicinity

    :param img: input 2D image
    :param m: rejection half-range
    :param nu: number of degrees of freedom in Student's distribution; `nu` = 0
        means infinity = Gaussian distribution, nu = 1 => Lorentzian
        distribution; also, `nu` = 2 and 4 are supported; for other values,
        CDF is not analytically invertible

    :return: mask image with 1-s corresponding to bad pixels
    """
    mask = np.zeros(img.shape, np.bool8)

    h, w = img.shape
    s = min(2*m + 1, w)
    for i in prange(h):
        for j in range(w):
            left = j - m
            d = w - left - s
            ofs = m
            if left < 0:
                ofs += left
                left = 0
            elif d < 0:
                ofs -= d
                left += d
            if chauvenet(
                    img[i, left:left+s], nu=nu, min_vals=2, mean_type=1,
                    sigma_type=1, check_idx=ofs)[0][ofs]:
                mask[i, j] = True

    return mask


@njit(nogil=True, parallel=True, cache=True)
def flag_columns(mask: np.ndarray) -> np.ndarray:
    """
    Flag pixels belonging to bad columns

    :param mask: mask with flagged outlying pixels, as returned by
        :func"`flag_horiz`

    :return: mask image with 1-s corresponding to bad pixels belonging to fully
        or partially bad columns
    """
    h, w = mask.shape

    # Identify columns with exceptionally high number of flagged pixels
    nrej_all = np.zeros(mask.shape[1], np.int32)
    for j in prange(mask.shape[1]):
        for i in range(mask.shape[0]):
            if mask[i, j]:
                nrej_all[j] += 1
    col_mask, mu, sigma = chauvenet(nrej_all, mean_type=1, sigma_type=1)
    flagged_col_indices = col_mask.nonzero()[0]
    n = len(flagged_col_indices)
    nrej = nrej_all[flagged_col_indices]
    min_hb = (h*(nrej - mu)/(h - mu)).astype(np.int32)  # min bad part height

    # Find the CDF for each bad column
    cdf = np.empty((n, h), np.int32)
    for i in prange(n):
        col = flagged_col_indices[i]
        x = cdf[i, 0] = int(mask[0, col])
        for row in range(1, h):
            if mask[row, col]:
                x += 1
            cdf[i, row] = x

    # Flag only pixels belonging to bad columns; find the maximum ranges of
    # rows at the bottom and the top of each bad column that are
    # not statistically outlying and will not be rejected
    mask[:] = False
    sqrt2 = np.sqrt(2)
    for p in prange(n):
        start = 0
        end = h - 1
        if nrej[p] > mu:
            best_hb = h
            for i in range(h - min_hb[p] + 1):  # start of bad part
                for j in range(i + min_hb[p] - 1, h - 1):  # end of bad part
                    hb = j - i + 1  # height of bad part of the column
                    if hb > best_hb:
                        continue
                    # Number of rejected pixels in the good part of the column
                    x = cdf[p, -1] - cdf[p, j]
                    if i:
                        x += cdf[p, i - 1]
                    hg = h - hb  # height of good part of the column
                    mu_prime = mu*hg/h
                    sigma_prime = sigma*hg/h*sqrt2
                    if math.erf(abs(x - mu_prime)/sigma_prime) < 1 - 0.5/hg:
                        if hb < best_hb:
                            best_hb = hb
                            start = i
                            end = j
                        elif j > end:
                            end = j

        mask[start:end+1, flagged_col_indices[p]] = True

    return mask


@njit(nogil=True, parallel=True, cache=True)
def flag_pixels(img: np.ndarray, col_mask: np.ndarray, m: int = 2,
                nu: int = 4) -> np.ndarray:
    """
    Reject outlying pixels not belonging to bad columns

    :param img: input 2D image
    :param col_mask: bad column mask as returned by :func:`flag_columns`
    :param m: bad column proximity half-range
    :param nu: number of degrees of freedom in Student's distribution; `nu` = 0
        means infinity = Gaussian distribution, nu = 1 => Lorentzian
        distribution; also, `nu` = 2 and 4 are supported; for other values,
        CDF is not analytically invertible

    :return: mask image with 1-s corresponding to bad pixels not belonging to
        bad columns
    """
    # Get indices of columns containing at least one flagged pixel
    bad_cols = np.zeros(col_mask.shape[1], np.bool8)
    for j in range(col_mask.shape[1]):
        for i in range(col_mask.shape[0]):
            if col_mask[i, j]:
                bad_cols[j] = True
                break
    bad_col_indices = bad_cols.nonzero()[0]

    h, w = img.shape
    mask = np.zeros_like(col_mask)
    for row in prange(h):
        for col in range(w):
            # Are we in a bad column?
            if col_mask[row, col]:
                continue

            # Detect proximity to bad column
            proximity_flag = False
            for j in bad_col_indices:
                if j - m <= col <= j + m:
                    proximity_flag = True
                    break

            # Extract (2*m + 1)x(2*m + 1) vicinity of the pixel not belonging
            # to bad columns
            pixel_set = np.empty((2*m + 1)**2, img.dtype)
            pixel_set[0] = img[row, col]
            npixels = 1
            for dy in range(-m, m + 1):
                i = row + dy
                if i < 0 or i >= h:
                    continue
                for dx in range(-m, m + 1):
                    if not dx and not dy:
                        # Central pixel already added
                        continue
                    j = col + dx
                    if j < 0 or j >= w or col_mask[i, j]:
                        continue
                    # Ignore non-central pixels in the same column in proximity
                    # to other bad columns
                    if not proximity_flag or j:
                        pixel_set[npixels] = img[i, j]
                        npixels += 1

            # For a special case where the pixel we are analyzing is surrounded
            # by bad columns and/or on the edge of the image
            if npixels < 2:
                for dy in range(-m, m + 1):
                    if not dy:
                        continue
                    i = row + dy
                    if 0 <= i < h and not col_mask[i, col]:
                        pixel_set[npixels] = img[i, col]
                        npixels += 1

            if chauvenet(
                    pixel_set[:npixels], nu=nu, mean_type=1, sigma_type=1,
                    min_vals=2, check_idx=0)[0][0]:
                mask[row, col] = True

    return mask


@njit(nogil=True, parallel=True, cache=True)
def correct_cols_and_pixels(
        img: np.ndarray, col_mask: np.ndarray, pixel_mask: np.ndarray,
        m_col: int = 2, m_pixel: int = 1) -> np.ndarray:
    """
    Replace cosmetic defects and/or cosmic rays by local average

    :param img: image to correct
    :param col_mask: bad column mask as returned by :func:`flag_columns`
    :param pixel_mask: isolated bad pixel mask as returned by
        :func:`flag_pixels`
    :param m_col: horizontal bad column averaging range
    :param m_pixel: bad pixel averaging range

    :return: corrected image
    """
    h, w = img.shape
    navg_col = 2*m_col
    navg_pixel = (2*m_pixel + 1)**2 - 1
    output = np.empty_like(img)
    for row in prange(h):
        for col in range(w):
            if col_mask[row, col]:
                # Replace bad column pixel by horizontal average of up to
                # navg_col non-flagged pixels
                avg_data = np.empty(navg_col, np.float64)
                navg = 0
                r = 1
                while navg < navg_col:
                    j = col - r
                    if j >= 0 and not col_mask[row, j] and \
                            not pixel_mask[row, j]:
                        avg_data[navg] = img[row, j]
                        navg += 1
                    if navg == navg_col:
                        break
                    j = col + r
                    if j < w and not col_mask[row, j] and \
                            not pixel_mask[row, j]:
                        avg_data[navg] = img[row, j]
                        navg += 1
                    r += 1
                    if r >= w:
                        break
                if navg:
                    output[row, col] = avg_data[:navg].mean()
                else:
                    # Not enough data for averaging, leave as is
                    output[row, col] = img[row, col]

            elif pixel_mask[row, col]:
                # Replace isolated bad pixel by average of up to navg_pixel
                # non-flagged pixels starting from 8 nearest neighbors
                avg_data = np.empty(navg_pixel, np.float64)
                navg = 0
                r = 1
                while navg < navg_pixel:
                    # Left edge
                    j = col - r
                    if j >= 0:
                        for dy in range(-r, r + 1):
                            i = row + dy
                            if 0 <= i < h and not col_mask[i, j] and \
                                    not pixel_mask[i, j]:
                                avg_data[navg] = img[i, j]
                                navg += 1
                                if navg == navg_pixel:
                                    break
                        if navg == navg_pixel:
                            break

                    # Top edge
                    i = row + r
                    if i < h:
                        for dx in range(-r, r + 1):
                            j = col + dx
                            if 0 <= j < w and not col_mask[i, j] and \
                                    not pixel_mask[i, j]:
                                avg_data[navg] = img[i, j]
                                navg += 1
                                if navg == navg_pixel:
                                    break
                        if navg == navg_pixel:
                            break

                    # Right edge
                    j = col + r
                    if j < w:
                        for dy in range(-r, r + 1):
                            i = row + dy
                            if 0 <= i < h and not col_mask[i, j] and \
                                    not pixel_mask[i, j]:
                                avg_data[navg] = img[i, j]
                                navg += 1
                                if navg == navg_pixel:
                                    break
                        if navg == navg_pixel:
                            break

                    # Bottom edge
                    i = row - r
                    if i >= 0:
                        for dx in range(-r, r + 1):
                            j = col + dx
                            if 0 <= j < w and not col_mask[i, j] and \
                                    not pixel_mask[i, j]:
                                avg_data[navg] = img[i, j]
                                navg += 1
                                if navg == navg_pixel:
                                    break
                        if navg == navg_pixel:
                            break

                    r += 1
                    if r >= w or r >= h:
                        break
                if navg:
                    output[row, col] = avg_data[:navg].mean()
                else:
                    # Not enough data for averaging, leave as is
                    output[row, col] = img[row, col]

            else:
                output[row, col] = img[row, col]

    return output


def correct_cosmetic(img: np.ndarray, m_col: int = 10, nu_col: int = 0,
                     m_pixel: int = 2, nu_pixel: int = 4,
                     m_corr_col: int = 2, m_corr_pixel: int = 1) \
        -> np.ndarray:
    """
    Fully automatic and self-contained intra-image cosmetic correction pipeline

    :param img: uncorrected 2D image
    :param m_col: bad column rejection half-range
    :param nu_col: number of degrees of freedom in Student's distribution for
        bad column rejection; `nu` = 0 means infinity = Gaussian distribution,
        nu = 1 => Lorentzian distribution; also, `nu` = 2 and 4 are supported;
        for other values, CDF is not analytically invertible
    :param m_pixel: bad column proximity half-range for isolated bad pixel
        rejection
    :param nu_pixel: number of degrees of freedom in Student's distribution for
        bad pixel rejection
    :param m_corr_col: bad column correction range
    :param m_corr_pixel: bad pixel correction range

    :return: corrected image
    """
    if img.dtype.name == 'float32':
        # Numba is slower for 32-bit floating point
        img = img.astype(np.float64)
    elif not img.dtype.isnative:
        # Non-native byte order is not supported by Numba
        img = img.byteswap().newbyteorder()
    initial_mask = flag_horiz(img, m=m_col, nu=nu_col)
    col_mask = flag_columns(initial_mask)
    pixel_mask = flag_pixels(img, col_mask, m=m_pixel, nu=nu_pixel)
    return correct_cols_and_pixels(
        img, col_mask, pixel_mask, m_col=m_corr_col, m_pixel=m_corr_pixel)
