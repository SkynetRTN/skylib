"""
Score functions for "lucky imaging" (optimal) stacking

Used by :func:`skylib.combine.stacking.combine`.
"""

from typing import Union

from numpy import ma, ndarray, sqrt

from ..calibration.background import estimate_background


__all__ = ['lucky_imaging_score']


def snr_score(img: Union[ndarray, ma.MaskedArray]) -> float:
    """
    Return image score based on the squared sum of per-pixel SNRs; used by
    :func:`combine` with `lucky_imaging` = "SNR"

    :param img: input image

    :return: scalar image score
    """
    bk, rms = estimate_background(img)
    return sqrt((((img - bk)/rms)**2).mean())


lucky_imaging_score = {
    'SNR': snr_score,
}
