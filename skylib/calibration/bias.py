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
    for i, hdu in enumerate(img):
        if not hdu.header.get('BIASCORR', False):
            # Apply each bias HDU to the corresponding image HDU; use the last
            # one if bias has less HDUs
            hdu.data -= bias[min(i, len(bias) - 1)].data
            hdu.header['BIASCORR'] = (
                os.path.basename(bias.filename()), 'Bias corrected')
