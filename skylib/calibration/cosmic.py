"""
Cosmic ray correction based on Astroscrappy

correct_cosmics(): remove cosmic rays from a FITS image
"""

from __future__ import division, print_function


__all__ = ['correct_cosmics']


def correct_cosmics(img, detect=False, sigclip=4.5, sigfrac=0.3, objlim=5.0,
                    gain=1.0, readnoise=10.0, satlevel=65535, niter=4,
                    sepmed=True, cleantype='meanmask', fsmode='median',
                    psfmodel='gauss', psffwhm=2.5, psfk=None, psfsize=7,
                    psfbeta=4.765):
    """
    Remove the tracks of cosmic rays from a FITS image using the Curtis
    McCully's Astroscrappy package based on the L.A.Cosmic algorithm by Pieter
    van Dokkum

    :param astropy.io.fits.HDUList img: input image; modified in place
    :param bool detect: return a cosmics-only image instead of a cleaned input
        image
    :param float sigclip: Laplacian-to-noise limit
    :param float sigfrac: fractional detection limit for neighboring pixels
    :param float objlim: minimum contrast between Laplacian image and fine
        structure image
    :param float gain: CCD inverse gain [e-/ADU]
    :param float readnoise: CCD readout noise [e-]
    :param float satlevel: CCD saturation level [ADU]
    :param int niter: number of iterations of L.A.Cosmic
    :param bool sepmed: use the separable median filter instead of the full one
    :param str cleantype: clean algorithm to use: "median", "medmask",
        "meanmask", "idw"
    :param str fsmode: method to build the fine structure image: "median" or
        "convolve"
    :param str psfmodel: model to use to generate the PSF kernel for fsmode ==
        "convolve" and psfk == None: "gauss", "gaussx", "gaussy", "moffat"
    :param float psffwhm: FWHM of the generated PSF [pixels]
    :param int psfsize: size of the generated PSF kernel [pixels]
    :param array_like psfk: PSF kernel array to use if fsmode == "convolve"; if
        None, use psfmodel, psffwhm, and psfsize to calculate the kernel
    :param float psfbeta: beta parameter for Moffat kernel

    :return: None
    """
    from numpy import uint8
    from astroscrappy import detect_cosmics

    for i, hdu in enumerate(img):
        if not hdu.header.get('CRCORR', False):
            crmask, cleanarr = detect_cosmics(
                hdu.data.astype(float), sigclip=sigclip, sigfrac=sigfrac,
                objlim=objlim, gain=gain, readnoise=readnoise,
                satlevel=satlevel, pssl=0.0, niter=niter, sepmed=sepmed,
                cleantype=cleantype, fsmode=fsmode, psfmodel=psfmodel,
                psffwhm=psffwhm, psfsize=psfsize, psfk=psfk, psfbeta=psfbeta)
            if detect:
                hdu.data = crmask.astype(uint8)
                hdu.header['CRONLY'] = (True, 'cosmic ray image')
            else:
                hdu.data = cleanarr
                hdu.header['CRCORR'] = (True, 'cosmics removed')
