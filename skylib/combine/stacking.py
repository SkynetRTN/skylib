"""
Image stacking.

:func:`~combine()`: combine a series of FITS images using the various stacking
modes with optional scaling and outlier rejection.
"""

from __future__ import absolute_import, division, print_function

from datetime import timedelta
import os.path
import logging

from numpy import (
    argmax, array, bincount, float32, indices, int32, ma, median, zeros)
import astropy.io.fits as pyfits

from ..util.stats import chauvenet
from ..util.fits import get_fits_time


__all__ = ['combine']


def combine(data, mode='average', scaling=None, rejection=None, min_keep=2,
            lo=None, hi=None):
    """
    Combine a series of FITS images using the various stacking modes with
    optional scaling and outlier rejection

    :param list[astropy.io.fits.HDUList] data: input datacube containing N FITS
        images of equal dimensions (n x m)
    :param str mode: stacking mode: "average" (default), "sum", or "median"
    :param str scaling: scaling mode: None (default) - do not scale data,
        "average" - scale data to match average values, "median" - match median
        values, "mode" - match modal values
    :param str rejection: outlier rejection mode: None (default) - do not reject
        outliers, "chauvenet" - use Chauvenet robust outlier rejection,
        "iraf" - IRAF-like clipping of `lo` lowest and `hi` highest values,
        "minmax" - reject values outside the absolute lower and upper limits
        (use with caution as `min_keep` below is not guaranteed, and you may end
        up in all values rejected for some or even all pixels), "sigclip" -
        iteratively reject pixels below and/or above the baseline
    :param int min_keep: minimum values to keep during rejection
    :param lo:
        `rejection` = "iraf": number of lowest values to clip; default: 1
        `rejection` = "minmax": reject values below this limit; default: not set
        `rejection` = "sigclip": reject values more than `lo` sigmas below the
            baseline; default: 3
    :param hi:
        `rejection` = "iraf": number of highest values to clip; default: 1;
        `rejection` = "minmax": reject values above this limit; default: not set
        `rejection` = "sigclip": reject values more than `hi` sigmas above the
            baseline; default: 3

    :return: FITS image with data set to the (n x m) combined array and header
        copied from one of the input images and modified to reflect the stacking
        mode and parameters
    :rtype: astropy.io.fits.HDUList
    """
    n = len(data)
    datacube = array([f[0].data for f in data]).astype(float32)

    # Scale data
    if scaling and n < 2:
        scaling = None
    if scaling:
        if scaling == 'average':
            k = datacube.mean((1, 2))
        elif scaling == 'median':
            k = median(datacube, (1, 2))
        elif scaling == 'mode':
            # Compute modal values from histograms; convert to integer and
            # assume 2 x 16-bit data range
            min_vals = datacube.min((1, 2))
            k = [
                argmax(bincount(d.ravel()))
                for d in (datacube - min_vals[..., None, None]).
                clip(0, 2*0x10000 - 1).astype(int32)] + min_vals
        else:
            raise ValueError('Unknown scaling mode "{}"'.format(scaling))

        # Normalize to the first frame; keep images with zero or same average
        # as is
        k[(k == 0).nonzero()] = 1
        k = k[0]/k
        scale_needed = (k != 1).nonzero()
        datacube[scale_needed] *= k[scale_needed][..., None, None]

    # Reject outliers
    rej_percent = 0.0
    if rejection and n < 2:
        rejection = None
    if rejection:
        datacube = ma.masked_array(datacube, zeros(datacube.shape, bool))

        if rejection == 'chauvenet':
            datacube.mask = chauvenet(datacube, min_vals=min_keep)
        elif rejection == 'iraf':
            if lo is None:
                lo = 1
            if hi is None:
                hi = 1
            if len(data) - (lo + hi) < min_keep:
                raise ValueError(
                    'IRAF rejection with lo={}, hi={} would keep less than '
                    '{} values for a {}-image set'.format(lo, hi, min_keep, n))
            if lo or hi:
                order = datacube.argsort(0)
                mg = indices(datacube.shape[1:])
                for j in range(-hi, lo):
                    datacube.mask[[order[j].ravel()] +
                                  [i.ravel() for i in mg]] = True
                del order, mg
        elif rejection == 'minmax':
            if lo is not None and hi is not None:
                if lo > hi:
                    raise ValueError(
                        'lo={} > hi={} for minmax rejection'.format(lo, hi))
                datacube.mask[((datacube < lo) |
                               (datacube > hi)).nonzero()] = True
                if datacube.mask.all(0).any():
                    logging.warn(
                        '%d completely masked pixels left after minmax '
                        'rejection', datacube.mask.all(0).sum())
        elif rejection == 'sigclip':
            if lo is None:
                lo = 3
            if hi is None:
                hi = 3
            if lo < 0 or hi < 0:
                raise ValueError(
                    'Lower and upper limits for sigma clipping must be '
                    'positive, got lo={}, hi={}'.format(lo, hi))
            max_rej = n - min_keep
            while True:
                avg = datacube.mean(0)
                sigma = datacube.std(0)
                resid = datacube - avg
                outliers = (datacube.mask.sum(0) < max_rej) & (sigma > 0) & (
                    (resid < -lo*sigma) | (resid > hi*sigma))
                if not outliers.any():
                    del avg, sigma, resid, outliers
                    break
                datacube.mask[outliers.nonzero()] = True
        else:
            raise ValueError('Unknown rejection mode "{}"'.format(rejection))

        if not datacube.mask.any():
            # Nothing was rejected
            datacube = datacube.data
        else:
            # Calculate the percentage of rejected pixels
            rej_percent = datacube.mask.sum()/datacube.size*100

    # Combine data
    if mode == 'average':
        res = datacube.mean(0)
    elif mode == 'sum':
        res = datacube.sum(0)
    elif mode == 'median':
        res = median(datacube, 0)
    else:
        raise ValueError('Unknown stacking mode "{}"'.format(mode))

    # Update FITS header, start from the first image
    hdr = data[0][0].header.copy(strip=True)

    exp_lengths = [
        im[0].header['EXPTIME'] if 'EXPTIME' in im[0].header and
        not isinstance(im[0].header['EXPTIME'], str)
        else im[0].header['EXPOSURE'] if 'EXPOSURE' in im[0].header and
        not isinstance(im[0].header['EXPOSURE'], str) else None
        for im in data]
    have_exp_lengths = exp_lengths.count(None) < len(exp_lengths)
    if have_exp_lengths:
        exp_lengths = array(
            [float(l) if l is not None else 0.0 for l in exp_lengths])
    t_start, t_cen, t_end = zip(*[get_fits_time(im[0].header) for im in data])

    hdr['FILTER'] = (','.join(
        {im[0].header['FILTER'] for im in data if 'FILTER' in im[0].header}),
        'Filter(s) used when taking images')

    hdr['OBSERVAT'] = (','.join(
        {im[0].header['OBSERVAT']
         for im in data if 'OBSERVAT' in im[0].header}),
        'Observatory or telescope name(s)')

    if have_exp_lengths:
        hdr['EXPTIME'] = hdr['EXPOSURE'] = (
            float(exp_lengths.sum() if mode == 'sum' else exp_lengths.mean()),
            '[s] Effective exposure time')

    try:
        hdr['DATE-OBS'] = (
            min([t for t in t_start if t is not None]),
            'Start time of the first exposure in stack')
    except ValueError:
        # No exposure start times
        pass

    if t_cen.count(None) < len(t_cen):
        # Calculate the average center time by converting to seconds since the
        # first exposure
        known_epochs = (
            array([i for i, t in enumerate(t_cen) if t is not None]),)
        t0 = min([t for t in t_cen if t is not None])
        epochs = array(
            [(t - t0).total_seconds() for t in t_cen if t is not None])
        if have_exp_lengths:
            total_exp = exp_lengths[known_epochs].sum()
        else:
            total_exp = 0
        if total_exp:
            # Weight center times by exposure lengths if the latter are known
            tc = (epochs*exp_lengths[known_epochs]).sum()/total_exp
        else:
            # Otherwise, use the average central time
            tc = epochs.mean()
        hdr['DATE-CEN'] = (
            (t0 + timedelta(seconds=tc)).isoformat(),
            'Weighted central time of image stack')

    try:
        hdr['DATE-END'] = (
            max([t for t in t_end if t is not None]),
            'Stop time of the last exposure in stack')
    except ValueError:
        # No exposure stop times
        pass

    hdr['COMBMETH'] = (mode.upper(), 'Combine method')

    hdr['REJMETH'] = (
        rejection.upper() if rejection is not None else 'NONE',
        'Rejection method used in combining')
    if rejection == 'iraf':
        hdr['NLOW'] = (lo, 'Number of low pixels rejected')
        hdr['NHIGH'] = (hi, 'Number of high pixels rejected')
    elif rejection == 'minmax':
        hdr['TLOW'] = (lo, 'Lower rejection threshold')
        hdr['THIGH'] = (hi, 'Upper rejection threshold')
    elif rejection == 'sigclip':
        hdr['SLOW'] = (lo, 'Lower sigma used with rejection')
        hdr['SHIGH'] = (hi, 'Upper sigma used with rejection')
    hdr['REJPRCNT'] = (float(rej_percent), 'Percentage of rejected pixels')

    hdr['SCAMETH'] = (
        scaling.upper() if scaling is not None else 'NONE',
        'Scale method used in combining')

    hdr['WGTMETH'] = ('NONE', 'Weight method used in combining')

    hdr['NCOMB'] = (n, 'Number of images used in combining')

    for i, im in enumerate(data):
        if im.filename():
            hdr['IMGS{:04d}'.format(i)] = (
                os.path.basename(im.filename()), 'Component filename')

    fits = pyfits.HDUList(pyfits.PrimaryHDU(
        data=res.data if isinstance(res, ma.MaskedArray) else res, header=hdr))
    return fits
