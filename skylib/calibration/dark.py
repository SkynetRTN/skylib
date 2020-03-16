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
    for i, hdu in enumerate(img):
        if not hdu.header.get('DARKCORR', False):
            dark_hdu = dark[min(i, len(dark) - 1)]
            t1, t2 = hdu.header['EXPTIME'], dark_hdu.header['EXPTIME']
            if not t2:
                raise ValueError('Invalid dark image. Exposure time = 0')
            scale = t1/t2
            if abs(scale - 1) > 1e-7:
                data = dark_hdu.data*scale
            else:
                data = dark_hdu.data
            hdu.data -= data
            hdu.header['DARKCORR'] = (
                os.path.basename(dark.filename()), 'Dark corrected')
            hdu.header['DARKSCAL'] = (scale, 'Scale factor for dark correction')
