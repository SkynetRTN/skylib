"""
Dark correction

correct_dark(): subtract a scaled dark from a FITS image
"""

from __future__ import absolute_import, division, print_function

import os.path


__all__ = ['correct_dark']


def correct_dark(img, dark):
    """
    Subtract dark from a FITS image with automatic scaling according to exposure
    time (EXPTIME keyword)

    :param astropy.io.fits.HDUList img: input image; modified in place
    :param astropy.io.fits.HDUList dark: dark image

    :return: None
    """
    t1, t2 = img[0].header['EXPTIME'], dark[0].header['EXPTIME']
    if not t2:
        raise ValueError('Invalid dark image. Exposure time = 0')
    scale = t1/t2
    if abs(scale - 1) > 1e-7:
        data = dark[0].data*scale
    else:
        data = dark[0].data
    img[0].data -= data
    img[0].header['DARKCORR'] = (
        os.path.basename(dark.filename()), 'Dark corrected')
    img[0].header['DARKSCAL'] = (scale, 'Scale factor for dark correction')
