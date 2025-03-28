"""
Automatic cropping.

:func:`~get_auto_crop()`: obtain the largest-area axis-aligned rectangle enclosed in the non-masked area of the image.

:func:`~get_edge_crop()`: obtain the smallest axis-aligned rectangle containing all non-masked pixels.
"""

from typing import Tuple

import numpy as np
from numba import njit


__all__ = ['get_auto_crop', 'get_edge_crop']


@njit(nogil=True, cache=True)
def max_rectangle(histogram: np.ndarray) -> Tuple[int, int, int]:
    """
    Find left/right boundaries and height of the largest rectangle that fits
    entirely under the histogram; see https://gist.github.com/zed/776423

    :param histogram: 1D non-negative integer array

    :return: left X coordinate, right X coordinate, and height of rectangle
    """
    stack = []
    left = right = height = pos = 0
    for pos, h in enumerate(histogram):
        start = pos
        while True:
            if not stack or h > stack[-1][1]:
                stack.append((start, h))
            elif stack and h < stack[-1][1]:
                top_start, top_height = stack[-1]
                if (pos - top_start + 1)*top_height > \
                        (right - left + 1)*height:
                    left, right, height = top_start, pos - 1, top_height
                start = stack.pop()[0]
                continue
            break

    for start, h in stack:
        if (pos - start + 1)*h > (right - left + 1)*height:
            left, right, height = start, pos, h

    return left, right, height


@njit(nogil=True, cache=True)
def get_auto_crop(mask: np.ndarray):
    """
    Obtain the largest-area axis-aligned rectangle enclosed in the non-masked
    area of the image; the algorithm is based on
    https://gist.github.com/zed/776423 and accelerated using Numba

    :param mask: 2D array with 1's corresponding to masked elements

    :return: cropping margins (left, right, bottom, top)
    """
    hist = (~(mask[0])).astype(np.uint32)
    left, right, rect_height = max_rectangle(hist)
    bottom = top = 0
    for i, row in enumerate(mask[1:]):
        for j, h in enumerate(row):
            if h:
                hist[j] = 0
            else:
                hist[j] += 1
        j1, j2, h = max_rectangle(hist)
        if (j2 - j1 + 1)*h > (right - left + 1)*rect_height:
            left, right, rect_height = j1, j2, h
            bottom, top = i + 2 - h, i + 1
    right = mask.shape[1] - 1 - right
    top = mask.shape[0] - 1 - top
    return left, right, bottom, top


def get_edge_crop(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Return the cropping margins (left, right, bottom, top) for the smallest axis-aligned rectangle containing all
    non-masked pixels of the image.

    :param mask: 2D array with 1's corresponding to masked elements

    :return: cropping margins (left, right, bottom, top)
    """
    h, w = mask.shape
    y, x = (~mask).nonzero()
    n = len(x)
    if not n:
        # All pixels are masked
        return 0, w, 0, h
    if n == w*h:
        # No masked pixels
        return 0, 0, 0, 0
    return x.min(), w - 1 - x.max(), y.min(), h - 1 - y.max()
