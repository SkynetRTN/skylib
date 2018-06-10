"""
Sky background subtraction using :class:`~sep.Background`

estimate_background(): extract background and noise map from an image.
"""

from __future__ import absolute_import, division, print_function

from numpy import atleast_1d, repeat, where
from numpy.ma import MaskedArray
import sep


__all__ = ['estimate_background', 'sep_compatible']


def sep_compatible(img):
    """
    Return data array compatible with SEP

    :param array_like img: input 2D image array

    :return: input array if compatible or its compatible view otherwise
    :rtype: numpy.ndarray
    """
    # Ensure native byte order
    if not img.dtype.isnative:
        img = img.byteswap().newbyteorder()

    # Convert to float32 unless the image is float64
    if img.dtype.char not in ('f', 'd'):
        img = img.astype('f')

    return img


def estimate_background(img, size=1/64, filter_size=3, fthresh=0.0):
    """
    Calculate background and noise maps

    This is a wrapper around :class:`sep.Background`.

    :param array_like img: NumPy array containing image data
    :param int | float | array_like size: box size for non-uniform background
        estimation: either one or two integer values in pixels or floating-point
        values from 0 to 1 in units of image size; for asymmetric box, Y size
        goes first
    :param int | array_like filter_size: window size of a 2D median filter to
        apply to the low-res background map; (ny, nx) or a single integer for
        ny = nx
    :param float fthresh: filter threshold

    :return: background and RMS maps as NumPy arrays of the same shape as input;
        for bkg_method="const", these are two scalars
    :rtype: tuple(array_like, array_like)
    """
    img = sep_compatible(img)

    size = atleast_1d(size)
    if len(size) == 1:
        size = repeat(size, 2)
    size = (where(
        size <= 1, size*img.shape, size) + 0.5).astype(int)
    bh, bw = size
    filter_size = atleast_1d(filter_size)
    if len(filter_size) == 1:
        filter_size = repeat(filter_size, 2)
    fh, fw = filter_size
    fthresh = float(fthresh)

    if isinstance(img, MaskedArray):
        mask = img.mask
        img = img.data
    else:
        mask = None
    bkg = sep.Background(
        img, mask=mask, bw=bw, bh=bh, fw=fw, fh=fh, fthresh=fthresh)
    return bkg.back(), bkg.rms()
