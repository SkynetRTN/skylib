"""
Sky background subtraction using :class:`~sep.Background`

estimate_background(): extract background and noise map from an image.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import sep


__all__ = ['estimate_background', 'sep_compatible']


def sep_compatible(img: np.ndarray | np.ma.MaskedArray) -> np.ndarray | np.ma.MaskedArray:
    """
    Return data array compatible with SEP

    :param img: input 2D image array

    :return: input array if compatible or its compatible view otherwise
    """
    # Ensure native byte order
    if not img.dtype.isnative:
        img = img.astype(img.dtype.newbyteorder())

    # Convert to float32 unless the image is float64
    if img.dtype.char not in ('f', 'd'):
        img = img.astype('f')

    return img


def estimate_background(img: np.ndarray | np.ma.MaskedArray,
                        size: int | float | tuple[int, int] | tuple[float, float] | tuple[int, float] |
                        tuple[float, int] = 1/64,
                        filter_size: int | tuple[int, int] = 3,
                        fthresh: float = 0.0) \
        -> tuple[np.ndarray, np.ndarray] | tuple[np.ma.MaskedArray, np.ma.MaskedArray]:
    """
    Calculate background and noise maps

    This is a wrapper around :class:`sep.Background`.

    :param array_like img: NumPy array containing image data
    :param int | float | array_like size: box size for non-uniform background estimation: either one or two integer
        values in pixels or floating-point values from 0 to 1 in units of image size; for asymmetric box, Y size goes
        first
    :param int | array_like filter_size: window size of a 2D median filter to apply to the low-res background map;
        (ny, nx) or a single integer for ny = nx
    :param float fthresh: filter threshold

    :return: background and RMS maps as NumPy arrays of the same shape as input; for `bkg_method`="const", these are two
        scalars
    """
    img = sep_compatible(img)

    size = np.atleast_1d(size)
    if len(size) == 1:
        size = np.repeat(size, 2)
    size = (np.where(size <= 1, size*img.shape, size) + 0.5).astype(int)
    bh, bw = size
    filter_size = np.atleast_1d(filter_size)
    if len(filter_size) == 1:
        filter_size = np.repeat(filter_size, 2)
    fh, fw = filter_size
    fthresh = float(fthresh)

    if isinstance(img, np.ma.MaskedArray):
        mask = img.mask
        img = img.data
    else:
        mask = None
    bkg = sep.Background(img, mask=mask, bw=bw, bh=bh, fw=fw, fh=fh, fthresh=fthresh)
    return bkg.back(), bkg.rms()
