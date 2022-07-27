"""
SkyLib image enhancement functions - wavelet-based algorithms.

wavelet_sharpen: a port of the Gimp wavelet sharpening plugin by Marco Rossini
"""


import numpy as np
from numba import njit


__all__ = ['wavelet_sharpen']


@njit(nogil=True, cache=True)
def hat_transform(base, st: int, size: int, sc: int):
    temp = np.zeros(size)
    for i in range(sc):
        temp[i] = 2*base[st*i] + base[st*(sc - i)] + base[st*(i + sc)]
    for i in range(sc, size - sc):
        temp[i] = 2*base[st*i] + base[st*(i - sc)] + base[st*(i + sc)]
    for i in range(size - sc, size):
        temp[i] = 2*base[st*i] + base[st*(i - sc)] + \
                  base[st*(2*size - 2 - (i + sc))]
    return temp


@njit(nogil=True, parallel=True, cache=True)
def wavelet_sharpen(img, amount: float = 0.1, radius: float = 0.5,
                    levels: int = 5):
    height, width = img.shape
    size = width*height
    out = img.ravel().astype(np.float64)
    fimg = [out, np.zeros(size), np.zeros(size)]

    hpass = lpass = 0
    for lev in range(levels):
        lpass = (lev & 1) + 1
        for row in range(height):
            temp = hat_transform(fimg[hpass][row*width:], 1, width, 1 << lev)
            for col in range(width):
                fimg[lpass][row*width + col] = temp[col]*0.25

        for col in range(width):
            temp = hat_transform(fimg[lpass][col:], width, height, 1 << lev)
            for row in range(height):
                fimg[lpass][row*width + col] = temp[row]*0.25

        amt = amount*np.exp(-(lev - radius)**2/1.5) + 1
        for i in range(size):
            fimg[hpass][i] -= fimg[lpass][i]
            fimg[hpass][i] *= amt

        if hpass:
            for i in range(size):
                fimg[0][i] += fimg[hpass][i]

        hpass = lpass

    for i in range(size):
        out[i] += fimg[lpass][i]

    return out.reshape((height, width))
