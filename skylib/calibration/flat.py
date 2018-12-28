"""
Flat correction

correct_flat(): divide a FITS image by an optionally normalized flat
"""

from __future__ import absolute_import, division, print_function

import os.path


__all__ = ['correct_flat']


def correct_flat(img, flat, normalize=True):
    """
    Divide a FITS image by flat optionally normalized to 1

    :param astropy.io.fits.HDUList img: input image; modified in place
    :param astropy.io.fits.HDUList flat: flat image
    :param bool normalize: normalize flat to 1 before division

    :return: None
    """
    for i, hdu in enumerate(img):
        if not hdu.header.get('FLATCORR', False):
            flat_hdu = flat[min(i, len(flat) - 1)]
            if normalize:
                avg = flat_hdu.data.mean()
                if not avg:
                    raise ValueError('Invalid flat image. Average count = 0')
                data = flat_hdu.data/avg
            else:
                data = flat_hdu.data
            hdu.data /= data
            hdu.header['FLATCORR'] = (
                os.path.basename(flat.filename()), 'Flat corrected')
