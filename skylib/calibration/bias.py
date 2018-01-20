"""
Bias correction

correct_bias(): subtract bias from a FITS image
"""

from __future__ import absolute_import, division, print_function

import os.path


__all__ = ['correct_bias']


def correct_bias(img, bias):
    """
    Subtract bias from a FITS image

    :param astropy.io.fits.HDUList img: input image; modified in place
    :param astropy.io.fits.HDUList bias: bias image

    :return: None
    """
    img[0].data -= bias[0].data
    img[0].header['BIASCORR'] = (
        os.path.basename(bias.filename()), 'Bias corrected')
