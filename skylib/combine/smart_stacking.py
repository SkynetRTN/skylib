"""
Score functions for smart stacking

Used by :func:`skylib.combine.stacking.combine`.
"""

from typing import Union

from numpy import array, ma, ndarray, sqrt
from scipy.ndimage import convolve

from ..calibration.background import estimate_background


__all__ = ['smart_stacking_score']


# Laplacian kernel
L = array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], float)/6


def snr_score(img: Union[ndarray, ma.MaskedArray]) -> float:
    """
    Return image score based on the total image SNR; used by :func:`combine`
    with `smart_stacking` = "SNR"

    :param img: input image

    :return: scalar image score
    """
    bk, rms = estimate_background(img)
    flux = (img - bk).sum()
    noise = flux + (rms**2).sum()
    if noise > 0:
        return flux/sqrt(noise)
    return 0


def sharpness_score(img: Union[ndarray, ma.MaskedArray]) -> float:
    """
    Return image score based on image sharpness; used by :func:`combine` with
    `smart_stacking` = "sharpness"

    :param img: input image

    :return: scalar image score
    """
    signal = img - estimate_background(img)[0]
    m = signal.mean()
    if m:
        return abs(convolve(signal/m, L, mode='nearest')).std()
    return 0


smart_stacking_score = {
    'SNR': snr_score,
    'sharpness': sharpness_score,
}
