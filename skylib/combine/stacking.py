"""
Image stacking.

:func:`~combine()`: combine a series of FITS images using the various stacking
modes with optional scaling and outlier rejection.
"""

from __future__ import absolute_import, division, print_function

from datetime import timedelta
import os.path
import logging
from typing import List, Optional, Tuple, Union

from numpy import (
    argmax, array, bincount, full, indices, int32, ma, median, nan,
    nanpercentile, ndarray, percentile as np_percentile)
import astropy.io.fits as pyfits

from ..util.stats import chauvenet
from ..util.fits import get_fits_time


__all__ = ['combine']


def combine(input_data: List[Union[pyfits.HDUList,
                                   Tuple[ndarray, pyfits.Header]]],
            mode: str = 'average', scaling: Optional[str] = None,
            rejection: Optional[str] = None, min_keep: int = 2,
            percentile: float = 50.0,
            lo: Optional[float] = None, hi: Optional[float] = None,
            max_mem_mb: float = 100.0, callback: Optional[callable] = None) \
        -> List[Tuple[ndarray, pyfits.Header]]:
    """
    Combine a series of FITS images using the various stacking modes with
    optional scaling and outlier rejection

    :param input_data: two or more FITS images or pairs (data, header)
        to combine; FITS files must be opened in readonly mode and have all
        the same number of HDUs and, separately for each HDU, the same data
        dimensions
    :param mode: stacking mode: "average" (default), "sum", or "percentile"
    :param scaling: scaling mode: None (default) - do not scale data,
        "average" - scale data to match average values, "percentile" - match
        the given percentile (median for `percentile` = 50), "mode" - match
        modal values
    :param rejection: outlier rejection mode: None (default) - do not reject
        outliers, "chauvenet" - use Chauvenet robust outlier rejection,
        "iraf" - IRAF-like clipping of `lo` lowest and `hi` highest values,
        "minmax" - reject values outside the absolute lower and upper limits
        (use with caution as `min_keep` below is not guaranteed, and you may end
        up in all values rejected for some or even all pixels), "sigclip" -
        iteratively reject pixels below and/or above the baseline
    :param min_keep: minimum values to keep during rejection
    :param percentile: for `mode`="percentile", default: 50 (median)
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
    :param max_mem_mb: maximum amount of RAM in megabytes to use during stacking
    :param callback: optional callable
            def callback(percent: float) -> None
        that is periodically called to update the progress of stacking operation

    :return: list of pairs (data, header) of the same length as the number
        of HDUs in the input FITS images (one if a (data, header) list
        was supplied on input), with data set to combined array(s) and header(s)
        copied from one of the input images and modified to reflect the stacking
        mode and parameters
    """
    n = len(input_data)
    if n < 2:
        raise ValueError('No data to combine')

    nhdus = None
    for f in input_data:
        if isinstance(f, pyfits.HDUList):
            m = len(f)
        else:
            m = 1
        if nhdus is None:
            nhdus = m
        elif m != nhdus:
            raise ValueError('All files must have the same number of HDUs')

    # Process each HDU separately
    output = []
    for hdu_no in range(nhdus):
        # Calculate scaling factors
        k_ref, k = None, []
        if scaling:
            for data_no, f in enumerate(input_data):
                if isinstance(f, pyfits.HDUList):
                    data = f[hdu_no].data
                else:
                    data = f[0]
                if scaling == 'average':
                    k.append(data.mean())
                elif scaling == 'percentile':
                    if percentile == 50:
                        k.append(
                            median(data) if not isinstance(data, ma.MaskedArray)
                            else ma.median(data))
                    else:
                        k.append(
                            np_percentile(data, percentile)
                            if not isinstance(data, ma.MaskedArray)
                            else np_percentile(data.compressed(), percentile))
                elif scaling == 'mode':
                    # Compute modal values from histograms; convert to integer
                    # and assume 2 x 16-bit data range
                    if isinstance(data, ma.MaskedArray):
                        data = data.compressed()
                    else:
                        data = data.ravel()
                    min_val = data.min(initial=0)
                    k.append(
                        argmax(bincount(
                            (data - min_val).clip(0, 2*0x10000 - 1)
                            .astype(int32))) + min_val)
                else:
                    raise ValueError(
                        'Unknown scaling mode "{}"'.format(scaling))
                if callback is not None:
                    callback((hdu_no + (data_no + 1)/n/2)/nhdus*100)

            # Normalize to the first frame with non-zero average; keep images
            # with zero or same average as is
            k_ref = k[0]
            if not k_ref:
                for ki in k[1:]:
                    if ki:
                        k_ref = ki
                        break

        # Process data in chunks to fit in the maximum amount of RAM allowed
        rowsize = 0
        data_width = data_height = 0
        for data in input_data:
            if isinstance(data, pyfits.HDUList):
                data = data[hdu_no].data
            else:
                data = data[0]
            h, w = data.shape
            if not rowsize:
                data_width, data_height = w, h
            elif (data_width, data_height) != (w, h):
                raise ValueError(
                    'Trying to combine arrays with non-matching dimensions: '
                    '{:d}x{:d} and {:d}x{:d}'.format(
                        data_width, data_height, w, h))
            rowsize += data[0].nbytes
            if rejection or isinstance(data, ma.MaskedArray):
                rowsize += data_width
        chunksize = min(max(int(max_mem_mb*(1 << 20)/rowsize), 1), data_height)
        while chunksize > 1:
            # Use as small chunks as possible but keep their total number
            if len(list(range(0, data_height, chunksize - 1))) > \
                    len(list(range(0, data_height, chunksize))):
                break
            chunksize -= 1
        chunks = []
        rej_percent = 0
        for chunk in range(0, data_height, chunksize):
            datacube = [
                f[hdu_no].data[chunk:chunk + chunksize]
                if isinstance(f, pyfits.HDUList)
                else f[0][chunk:chunk + chunksize]
                for f in input_data
            ]
            if k_ref:
                # Scale data
                for data, ki in zip(datacube, k):
                    if ki not in (0, k_ref):
                        data *= k_ref/ki

            # Reject outliers
            if rejection or any(isinstance(data, ma.MaskedArray)
                                for data in datacube):
                datacube = ma.masked_array(datacube)
                if not datacube.mask.shape:
                    # No initially masked data, but we'll need an array instead
                    # of mask=False to do slicing operations
                    datacube.mask = full(datacube.shape, datacube.mask)
            else:
                datacube = array(datacube)

            if rejection == 'chauvenet':
                datacube.mask = chauvenet(datacube, min_vals=min_keep)
            elif rejection == 'iraf':
                if lo is None:
                    lo = 1
                if hi is None:
                    hi = 1
                if n - (lo + hi) < min_keep:
                    raise ValueError(
                        'IRAF rejection with lo={}, hi={} would keep less than '
                        '{} values for a {}-image set'.format(
                            lo, hi, min_keep, n))
                if lo or hi:
                    # Mask "lo" smallest values and "hi" largest values along
                    # the 0th axis
                    order = datacube.argsort(0)
                    mg = tuple(i.ravel() for i in indices(datacube.shape[1:]))
                    for j in range(-hi, lo):
                        datacube.mask[(order[j].ravel(),) + mg] = True
                    del order, mg
            elif rejection == 'minmax':
                if lo is not None and hi is not None:
                    if lo > hi:
                        raise ValueError(
                            'lo={} > hi={} for minmax rejection'.format(lo, hi))
                    datacube.mask[((datacube < lo) |
                                   (datacube > hi)).nonzero()] = True
                    if datacube.mask.all(0).any():
                        logging.warning(
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
                    outliers = (datacube.mask.sum(0) < max_rej) & \
                        (sigma > 0) & ((resid < -lo*sigma) | (resid > hi*sigma))
                    if not outliers.any():
                        del avg, sigma, resid, outliers
                        break
                    datacube.mask[outliers.nonzero()] = True
            elif rejection:
                raise ValueError(
                    'Unknown rejection mode "{}"'.format(rejection))

            if isinstance(datacube, ma.MaskedArray):
                if datacube.mask is None or not datacube.mask.any():
                    # Nothing was rejected
                    datacube = datacube.data
                else:
                    # Calculate the percentage of rejected pixels
                    rej_percent += datacube.mask.sum()

            # Combine data
            if mode == 'average':
                res = datacube.mean(0)
            elif mode == 'sum':
                res = datacube.sum(0)
            elif mode == 'percentile':
                if percentile == 50:
                    if isinstance(datacube, ma.MaskedArray):
                        res = ma.median(datacube, 0)
                    else:
                        res = median(datacube, 0)
                else:
                    if isinstance(datacube, ma.MaskedArray):
                        res = nanpercentile(datacube.filled(nan), percentile, 0)
                    else:
                        res = np_percentile(datacube, percentile, 0)
            else:
                raise ValueError('Unknown stacking mode "{}"'.format(mode))
            chunks.append(res)

            if callback is not None:
                callback(
                    (hdu_no + (0.5 if scaling else 0) +
                     min(chunk + chunksize, data_height)/data_height /
                     (2 if scaling else 1))/nhdus*100)

        if len(chunks) > 1:
            res = ma.vstack(chunks)
        else:
            res = chunks[0]
        if isinstance(res, ma.MaskedArray) and (
                res.mask is None or not res.mask.any()):
            res = res.data
        del chunks

        # Update FITS headers, start from the first image
        headers = [f[hdu_no].header if isinstance(f, pyfits.HDUList) else f[1]
                   for f in input_data]
        hdr = headers[0].copy(strip=True)

        exp_lengths = [
            h['EXPTIME'] if 'EXPTIME' in h and not isinstance(h['EXPTIME'], str)
            else h['EXPOSURE'] if 'EXPOSURE' in h and
            not isinstance(h['EXPOSURE'], str) else None
            for h in headers]
        have_exp_lengths = exp_lengths.count(None) < len(exp_lengths)
        if have_exp_lengths:
            exp_lengths = array(
                [float(l) if l is not None else 0.0 for l in exp_lengths])
        t_start, t_cen, t_end = tuple(zip(*[get_fits_time(h) for h in headers]))

        hdr['FILTER'] = (','.join(
            {h['FILTER'] for h in headers if 'FILTER' in h}),
            'Filter(s) used when taking images')

        hdr['OBSERVAT'] = (','.join(
            {h['OBSERVAT'] for h in headers if 'OBSERVAT' in h}),
            'Observatory or telescope name(s)')

        if have_exp_lengths:
            hdr['EXPTIME'] = hdr['EXPOSURE'] = (
                float(exp_lengths.sum() if mode == 'sum'
                      else exp_lengths.mean()), '[s] Effective exposure time')

        try:
            hdr['DATE-OBS'] = (
                min([t for t in t_start if t is not None]),
                'Start time of the first exposure in stack')
        except ValueError:
            # No exposure start times
            pass

        if t_cen.count(None) < len(t_cen):
            # Calculate the average center time by converting to seconds since
            # the first exposure
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
                # Weight center times by exposure lengths if the latter are
                # known
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
        hdr['REJPRCNT'] = (float(rej_percent/data_width/data_height/n*100),
                           'Percentage of rejected pixels')

        hdr['SCAMETH'] = (
            scaling.upper() if scaling is not None else 'NONE',
            'Scale method used in combining')

        hdr['WGTMETH'] = ('NONE', 'Weight method used in combining')

        hdr['NCOMB'] = (n, 'Number of images used in combining')

        for i, im in enumerate(input_data):
            if isinstance(im, pyfits.HDUList) and im.filename():
                hdr['IMGS{:04d}'.format(i)] = (
                    os.path.basename(im.filename()), 'Component filename')

        output.append((res, hdr))

    return output
