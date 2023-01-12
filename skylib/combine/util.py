"""
Helper functions for image combining
"""

from typing import Optional, Tuple, Union

import numpy as np
from numpy import ma
import astropy.io.fits as pyfits


__all__ = ['get_data']


def get_data(f: Union[pyfits.HDUList, Tuple[Union[np.ndarray, ma.MaskedArray],
                                            pyfits.Header]],
             hdu_no: int,
             start: int = 0,
             end: Optional[int] = None,
             downsample: int = 1) -> Union[np.ndarray, ma.MaskedArray]:
    """
    Return data array given stacking input data item (either a FITS file or
    an array+header); handles masks and NaNs and optionally does software
    binning (downsampling)

    :param f: input data item, as passed to :func:`combine` as `input_data`
    :param hdu_no: optional FITS HDU number if applicable
    :param start: starting row of data to load
    :param end: ending data row to load
    :param downsample: software binning factor

    :return: data array (masked or unmasked)
    """
    if isinstance(f, pyfits.HDUList):
        data = f[hdu_no].data[start:end]
    else:
        data = f[0][start:end]
    if data.dtype.kind != 'f':
        data = data.astype(float)
    nans = np.isnan(data)
    if nans.any():
        if not isinstance(data, ma.MaskedArray):
            data = ma.masked_array(data, np.zeros_like(data, bool))
        data.mask[nans] = True
    if downsample < 2:
        return data

    # Software binning
    h, w = data.shape
    width = w//downsample
    height = h//downsample
    if h/downsample % 1:
        data = data[:height*downsample]
    if w/downsample % 1:
        data = data[:, :width*downsample]
    if not isinstance(data, ma.MaskedArray) or not np.shape(data.mask):
        return (
                data.reshape(height, downsample, width, downsample)
                .sum(3).sum(1)/downsample**2).astype(data.dtype)
    return ma.masked_array(
        (data.data.reshape(height, downsample, width, downsample)
         .sum(3).sum(1)/downsample**2).astype(data.dtype),
        data.mask.reshape(height, downsample, width, downsample)
        .sum(3).sum(1)/downsample**2)
