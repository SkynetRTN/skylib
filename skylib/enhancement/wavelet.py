"""
SkyLib image enhancement functions - wavelet-based algorithms.

wavelet_sharpen: a port of the Gimp wavelet sharpening plugin by Marco Rossini
"""


import numpy as np
from numba import njit, prange


__all__ = ['wavelet_sharpen']


@njit(nogil=True, cache=True)
def hat_transform_row(base: np.ndarray, sc: int) -> np.ndarray:
    size = base.size
    temp = np.zeros(size)
    for i in prange(sc):
        temp[i] = 2*base[i] + base[sc - i] + base[i + sc]
    for i in prange(sc, size - sc):
        temp[i] = 2*base[i] + base[i - sc] + base[i + sc]
    ofs = 2*(size - 1) - sc
    for i in prange(size - sc, size):
        temp[i] = 2*base[i] + base[i - sc] + base[ofs - i]
    return temp


@njit(nogil=True, cache=True)
def hat_transform_col(base: np.ndarray, st: int, size: int, sc: int) \
        -> np.ndarray:
    temp = np.zeros(size)
    st_sc = st*sc
    for i in prange(0, sc):
        st_i = st*i
        temp[i] = 2*base[st_i] + base[st_sc - st_i] + base[st_i + st_sc]
    for i in prange(sc, size - sc):
        st_i = st*i
        temp[i] = 2*base[st_i] + base[st_i - st_sc] + base[st_i + st_sc]
    ofs = 2*st*(size - 1) - st_sc
    for i in prange(size - sc, size):
        st_i = st*i
        temp[i] = 2*base[st_i] + base[st_i - st_sc] + base[ofs - st_i]
    return temp


@njit(nogil=True, parallel=True, cache=True)
def _wavelet_sharpen(img, amount: float, radius: float, levels: int) \
        -> np.ndarray:
    height, width = img.shape
    size = width*height
    out = img.ravel().copy()
    fimg = [out, np.zeros(size), np.zeros(size)]

    hpass = lpass = 0
    for lev in range(levels):
        lpass = (lev & 1) + 1
        sc = 1 << lev
        for row in prange(height):
            i0 = row*width
            temp = hat_transform_row(fimg[hpass][i0:i0 + width], sc)
            for col in range(width):
                fimg[lpass][i0 + col] = temp[col]*0.25

        for col in prange(width):
            temp = hat_transform_col(fimg[lpass][col:], width, height, sc)
            for row in range(height):
                fimg[lpass][row*width + col] = temp[row]*0.25

        amt = amount*np.exp(-(lev - radius)**2/1.5) + 1
        for i in prange(size):
            fimg[hpass][i] -= fimg[lpass][i]
            fimg[hpass][i] *= amt

        if lev:
            for i in prange(size):
                out[i] += fimg[hpass][i]

        hpass = lpass

    for i in prange(size):
        out[i] += fimg[lpass][i]

    return out.reshape((height, width))


def wavelet_sharpen(img: np.ndarray, amount: float = 0.5,
                    radius: float = 0.5, levels: int = 5) -> np.ndarray:
    """
    Wavelet image sharpening

    :param img: input 2D image
    :param amount: sharpening amount, > 0
    :param radius: sharpening radius, > 0
    :param levels: number of wavelet levels

    :return: sharpened image, dtype = float64
    """
    height, width = img.shape
    if (1 << levels) > min(width, height):
        raise ValueError(
            'Number of wavelet levels too large for this image size')
    return _wavelet_sharpen(img.astype(np.float64), amount, radius, levels)
