"""
Score functions for smart stacking

Used by :func:`skylib.combine.stacking.combine`.
"""

from typing import Union

from numpy import ma, ndarray, sqrt

from ..calibration.background import estimate_background


__all__ = ['smart_stacking_score']


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


smart_stacking_score = {
    'SNR': snr_score,
}
