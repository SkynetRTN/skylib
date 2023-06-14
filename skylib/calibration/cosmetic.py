"""
Numba-accelerated cosmetic defect correction

Adapted from the original code by Logan Selph.

:func:`correct_cosmetic`: high-level automatic intra-image correction pipeline.
:func:`flag_horiz`, :func:`flag_columns`, :func:`flag_pixels`: individual
    cosmetic defect rejection steps.
:func:`detect_defects`: obtain bad column and pixel masks from the given image;
    a wrapper around :func:`flag_*`
:func:`correct_cols_and_pixels`: eliminate cosmetic defects by local averaging
    using rejection maps obtained by :func:`flag_*` or :func:`detect_defects`.
:func:`correct_cosmetic`: obtain and eliminate cosmetic defects from
    a single image or apply the previously obtained bad column and pixel masks
"""

import math
from typing import Optional, Tuple

import numpy as np
from numba import njit, prange

from ..util.stats import chauvenet1


__all__ = [
    # High-level interface
    'detect_defects', 'correct_cosmetic',
    # Internal lower-level defect detection and correction functions
    'flag_horiz', 'flag_columns', 'flag_pixels', 'correct_cols_and_pixels',
]


@njit(nogil=True, parallel=True, cache=True)
def flag_horiz(img: np.ndarray, m: int = 10, nu: int = 0, z: int = 1) \
        -> np.ndarray:
    """
    Flag pixels with outlying values within their [-m,m] horizontal vicinity

    :param img: input 2D image
    :param m: rejection half-range
    :param nu: number of degrees of freedom in Student's distribution; `nu` = 0
        means infinity = Gaussian distribution, nu = 1 => Lorentzian
        distribution; also, `nu` = 2 and 4 are supported; for other values,
        CDF is not analytically invertible
    :param z: number of binning iterations

    :return: mask image with 1-s corresponding to bad pixels
    """
    h, w = img.shape
    s = min(2*m + 1, w)
    mask = np.zeros(img.shape, np.bool8)
    if z > 1:
        binned_img = img.copy()  # will modify binned image in place
    else:
        binned_img = img

    if nu == 1:
        q = 0.5
    elif nu == 2:
        q = 0.577
    elif nu == 4:
        q = 0.626
    else:
        q = 0.683
    q += 1e-7

    for n in range(z):
        if n:
            # Bin the input array
            for i in prange(h):
                for j in range(w):
                    x1 = binned_img[2*i, j]
                    x2 = binned_img[2*i+1, j]
                    if np.isfinite(x1):
                        if np.isfinite(x2):
                            binned_img[i, j] = (x1 + x2)/2
                        else:
                            binned_img[i, j] = x1
                    elif np.isfinite(x2):
                        binned_img[i, j] = x2
                    else:
                        binned_img[i, j] = np.nan

        for i in prange(h):
            for j in range(w):
                if n and not np.isfinite(binned_img[i, j]):
                    # No data for the current pixel, already masked at
                    # the previous iterations
                    continue

                left = j - m
                d = w - left - s
                ofs = m
                if left < 0:
                    ofs += left
                    left = 0
                elif d < 0:
                    ofs -= d
                    left += d
                rej = np.zeros(s, np.bool8)
                chauvenet1(
                    binned_img[i, left:left+s], rej, nu=nu, min_vals=2,
                    mean_type=1, mean_override=None, sigma_type=1,
                    sigma_override=None, clip_lo=True, clip_hi=True,
                    max_iter=0, check_idx=ofs, q=q)
                if rej[ofs]:
                    # Mask the whole binned pixel
                    if z > 1:
                        binned_img[i, j] = np.nan
                    if n:
                        r = 1 << n
                        row = i*r
                        for k in range(r):
                            mask[row+k, j] = True
                    else:
                        mask[i, j] = True

        h //= 2  # prepare to the next binning iteration

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
    col_mask = np.zeros(nrej_all.shape, np.bool8)
    mu, sigma = chauvenet1(
        nrej_all, col_mask, nu=0, min_vals=10, mean_type=1, mean_override=None,
        sigma_type=1, sigma_override=None, clip_lo=False, clip_hi=True,
        max_iter=1, check_idx=None, q=0.6830001)[1:]
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
    for j in prange(col_mask.shape[1]):
        for i in range(col_mask.shape[0]):
            if col_mask[i, j]:
                bad_cols[j] = True
                break
    bad_col_indices = bad_cols.nonzero()[0]

    if nu == 1:
        q = 0.5
    elif nu == 2:
        q = 0.577
    elif nu == 4:
        q = 0.626
    else:
        q = 0.683
    q += 1e-7

    h, w = img.shape
    mask = np.zeros_like(col_mask)
    for row in prange(h):
        for col in range(w):
            # Are we in a bad column?
            if col_mask[row, col]:
                continue

            # Extract (2*m + 1)x(2*m + 1) vicinity of the pixel not belonging
            # to bad columns
            pixel_set = np.empty((2*m + 1)**2, img.dtype)
            pixel_set[0] = img[row, col]
            npixels = 1
            for j in range(max(col - m, 0), min(col + m + 1, w)):
                # Ignore pixels in bad columns even if they are not marked
                # as bad themselves
                if j in bad_col_indices:
                    continue

                for i in range(max(row - m, 0), min(row + m + 1, h)):
                    if i != row or j != col:
                        pixel_set[npixels] = img[i, j]
                        npixels += 1

            # For a special case where the pixel we are analyzing is surrounded
            # by bad columns and/or on the edge of the image, use only
            # non-masked pixels above and below the current one
            if npixels < 3:
                npixels = 1
                for i in range(max(row - m, 0), min(row + m + 1, h)):
                    if i != row and not col_mask[i, col]:
                        pixel_set[npixels] = img[i, col]
                        npixels += 1

            # Mark the pixel as bad if it's outlying compared to its non-masked
            # vicinity or if there are not enough non-masked pixels to make
            # a decision
            if npixels < 3:
                mask[row, col] = True
            else:
                rej = np.zeros(npixels, np.bool8)
                if chauvenet1(
                        pixel_set[:npixels], rej, nu=nu, min_vals=2,
                        mean_type=1, mean_override=None, sigma_type=1,
                        sigma_override=None, clip_lo=True, clip_hi=True,
                        max_iter=0, check_idx=0, q=q)[0][0]:
                    mask[row, col] = True

    return mask


def detect_defects(img: np.ndarray, m_col: int = 10, nu_col: int = 0,
                   z: int = 1, m_pixel: int = 2, nu_pixel: int = 4) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect cosmetic defects (bad columns and isolated pixels) from the given
    image

    :param img: input image
    :param m_col: bad column detection half-range
    :param nu_col: number of degrees of freedom in Student's distribution for
        bad column detection; `nu` = 0 means infinity = Gaussian distribution,
        nu = 1 => Lorentzian distribution; also, `nu` = 2 and 4 are supported;
        for other values, CDF is not analytically invertible
    :param z: number of binning iterations
    :param m_pixel: bad column proximity half-range for isolated bad pixel
        detection
    :param nu_pixel: number of degrees of freedom in Student's distribution for
        bad pixel detection

    :return: bad column and bad pixel masks that can be passed to
        :func:`correct_cols_and_pixels` or :func:`correct_cosmetic`
    """
    if img.dtype.name != 'float64':
        img = img.astype(np.float64)
    if not img.dtype.isnative:
        # Non-native byte order is not supported by Numba
        img = img.byteswap().newbyteorder()

    col_mask = flag_columns(flag_horiz(img, m=m_col, nu=nu_col, z=z))
    pixel_mask = flag_pixels(img, col_mask, m=m_pixel, nu=nu_pixel)

    return col_mask, pixel_mask


@njit(nogil=True, parallel=True, cache=True)
def correct_cols_and_pixels(
        img: np.ndarray, col_mask: np.ndarray, pixel_mask: np.ndarray,
        m_col: int = 2, m_pixel: int = 1) -> np.ndarray:
    """
    Replace cosmetic defects and/or cosmic rays by local average

    WARNING. This function requires a specific dtype of `img`; use
             :func:`correct_cosmetic` if `img`.dtype is not guaranteed.

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
                    output[row, col] = np.median(avg_data[:navg])
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
                    output[row, col] = np.median(avg_data[:navg])
                else:
                    # Not enough data for averaging, leave as is
                    output[row, col] = img[row, col]

            else:
                output[row, col] = img[row, col]

    return output


def correct_cosmetic(
        img: np.ndarray, col_mask: Optional[np.ndarray] = None,
        pixel_mask: Optional[np.ndarray] = None, m_col: int = 10,
        nu_col: int = 0, z: int = 1, m_pixel: int = 2, nu_pixel: int = 2,
        m_corr_col: int = 2, m_corr_pixel: int = 1) -> np.ndarray:
    """
    Fully automatic and self-contained intra-image cosmetic correction pipeline

    :param img: uncorrected 2D image
    :param col_mask: optional bad column mask; if omitted, estimated on the fly
        from `img` using :func:`flag_horiz` and :func:`flag_columns`
    :param pixel_mask: optional isolated bad pixel mas; if omitted, estimated
        on the fly from the image using :func:`flag_pixels`
    :param m_col: bad column detection half-range
    :param nu_col: number of degrees of freedom in Student's distribution for
        bad column detection; `nu` = 0 means infinity = Gaussian distribution,
        nu = 1 => Lorentzian distribution; also, `nu` = 2 and 4 are supported;
        for other values, CDF is not analytically invertible
    :param z: number of binning iterations
    :param m_pixel: bad column proximity half-range for isolated bad pixel
        detection
    :param nu_pixel: number of degrees of freedom in Student's distribution for
        bad pixel detection
    :param m_corr_col: bad column correction range
    :param m_corr_pixel: bad pixel correction range

    :return: corrected image
    """
    if img.dtype.name != 'float64':
        img = img.astype(np.float64)
    if not img.dtype.isnative:
        # Non-native byte order is not supported by Numba
        img = img.byteswap().newbyteorder()

    if col_mask is None:
        col_mask = flag_columns(flag_horiz(img, m=m_col, nu=nu_col, z=z))
    else:
        if col_mask.shape != img.shape:
            raise ValueError(
                f'Bad column map shape mismatch: expected '
                f'{img.shape[1]}x{img.shape[0]}, got '
                f'{col_mask.shape[1]}x{col_mask.shape[0]}')
        if col_mask.dtype.name != 'bool':
            col_mask = col_mask.astype(np.bool8)

    if pixel_mask is None:
        pixel_mask = flag_pixels(img, col_mask, m=m_pixel, nu=nu_pixel)
    else:
        if pixel_mask.shape != img.shape:
            raise ValueError(
                f'Bad pixel map shape mismatch: expected '
                f'{img.shape[1]}x{img.shape[0]}, got '
                f'{col_mask.shape[1]}x{col_mask.shape[0]}')
        if pixel_mask.dtype.name != 'bool':
            pixel_mask = pixel_mask.astype(np.bool8)

    return correct_cols_and_pixels(
        img, col_mask, pixel_mask, m_col=m_corr_col, m_pixel=m_corr_pixel)
