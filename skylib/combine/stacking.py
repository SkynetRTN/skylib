"""
Image stacking.

:func:`~combine()`: combine a series of FITS images using the various stacking
modes with optional scaling and outlier rejection.
"""

import gc
import os.path
import logging
from typing import List, Optional, Tuple, Union
from datetime import timedelta

import numpy as np
from numpy import ma
import astropy.io.fits as pyfits

from ..util.stats import chauvenet
from ..util.fits import get_fits_time
from .mosaicing import get_equalization_transforms, global_equalize
from .smart_stacking import smart_stacking_score
from .util import get_data


__all__ = ['combine']


def _do_combine(hdu_no: int, progress: float, progress_step: float,
                data_width: int, data_height: int,
                input_data: List[Union[pyfits.HDUList,
                                       Tuple[Union[np.ndarray, ma.MaskedArray],
                                             pyfits.Header]]],
                mode: str = 'average', scaling: Optional[str] = None,
                rejection: Optional[str] = None, min_keep: int = 2,
                propagate_mask: bool = True, percentile: float = 50.0,
                lo: Optional[float] = None, hi: Optional[float] = None,
                equalize_additive: bool = False, equalize_order: int = 1,
                equalize_multiplicative: bool = True,
                equalize_global: bool = False,
                max_mem_mb: float = 100.0,
                callback: Optional[callable] = None) \
        -> Tuple[Union[np.ndarray, ma.MaskedArray], float]:
    """
    Combine the given HDUs from all input images; used by :func:`combine` to
    get a stack of either all input images or, if smart stacking is enabled,
    of their subset

    :return: image stack data and rejection percent
    """
    n = len(input_data)

    k_ref, k, offsets, transformations = 1, [1]*n, [0]*n, {}
    if equalize_additive or equalize_multiplicative:
        # Equalize the common image parts; suitable for mosaicing
        transformations = get_equalization_transforms(
            hdu_no, progress, progress_step, data_width, data_height,
            input_data, equalize_additive, equalize_order,
            equalize_multiplicative, max_mem_mb, callback)

    if scaling:
        # Calculate offsets and scaling factors
        for data_no, f in enumerate(input_data):
            data = get_data(f, hdu_no)

            if scaling == 'average':
                avg = data.mean()

            elif scaling == 'percentile':
                if percentile == 50:
                    avg = np.median(data) \
                        if not isinstance(data, ma.MaskedArray) \
                        else ma.median(data)
                else:
                    avg = np.percentile(data, percentile) \
                        if not isinstance(data, ma.MaskedArray) \
                        else np.percentile(data.compressed(), percentile)

            elif scaling == 'mode':
                # Compute modal values from histograms; convert to integer
                # and assume 2 x 16-bit data range
                if isinstance(data, ma.MaskedArray):
                    data = data.compressed()
                else:
                    data = data.ravel()
                min_val = data.min(initial=0)
                avg = np.argmax(np.bincount(
                    (data - min_val).clip(0, 2*0x10000 - 1)
                    .astype(np.int32))) + min_val

            else:
                raise ValueError(
                    'Unknown scaling mode "{}"'.format(scaling))

            if avg > 0:
                ofs = 0
            else:
                # To make sure that all images are scaled by a positive
                # factor, add a constant offset to the image so that its
                # average = 1
                ofs, avg = 1 - avg, 1
            k[data_no] = avg
            offsets[data_no] = ofs

            gc.collect()
            if callback is not None:
                callback(progress + (data_no + 1)/n/2*progress_step)

        # Normalize to the first frame with positive average
        k_ref = k[0]
        if k_ref == 1:
            for ki in k[1:]:
                if ki != 1:
                    k_ref = ki
                    break

    # Process data in chunks to fit in the maximum amount of RAM allowed
    bytes_per_pixel = max(
        abs((data[hdu_no].header if isinstance(data, pyfits.HDUList)
             else data[1])['BITPIX'])//8 for data in input_data) + 1
    rowsize = data_width*bytes_per_pixel*len(input_data)
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
            get_data(f, hdu_no, chunk, chunk + chunksize)
            for f in input_data
        ]

        # Scale data
        for data, ki, ofsi in zip(datacube, k, offsets):
            if ofsi:
                data += ofsi
            if ki != k_ref:
                data *= k_ref/ki

        # Equalize backgrounds
        if transformations:
            if equalize_additive:
                chunk_y, chunk_x = np.indices(datacube[0].shape)
                chunk_y += chunk
                x_pow, y_pow = [1], [1]
                if equalize_order > 0:
                    x_pow.append(chunk_x)
                    y_pow.append(chunk_y)
                    for p in range(equalize_order - 1):
                        x_pow.append(x_pow[-1]*chunk_x)
                        y_pow.append(y_pow[-1]*chunk_y)
                del chunk_x, chunk_y
            else:
                x_pow = y_pow = None
            for i, coeffs in transformations.items():
                data = datacube[i]
                pofs = 0
                if equalize_multiplicative and i:
                    data *= coeffs[pofs]
                    pofs += 1
                if equalize_additive:
                    for o in range(equalize_order + 1):
                        for xp in range(o, -1, -1):
                            c = coeffs[pofs]
                            if c:
                                yp = o - xp
                                if xp:
                                    if yp:
                                        d = x_pow[xp]*y_pow[yp]
                                    else:
                                        d = x_pow[xp]
                                else:
                                    d = y_pow[yp]
                                data += c*d
                            pofs += 1
            del x_pow, y_pow
            gc.collect()

        initial_mask = None
        if rejection or any(isinstance(data, ma.MaskedArray)
                            for data in datacube):
            datacube = ma.masked_array(datacube, fill_value=np.nan)
            if not datacube.mask.shape:
                # No initially masked data, but we'll need an array instead
                # of mask=False to do slicing operations
                datacube.mask = np.full(datacube.shape, datacube.mask)

            if propagate_mask:
                # After stacking (and possibly rejection), we'll mask all
                # pixels that are initially masked in at least one image
                # (e.g. edges/corners after alignment)
                initial_mask = np.logical_or.reduce(datacube.mask, axis=0)
                if not initial_mask.any():
                    initial_mask = None
        else:
            datacube = np.array(datacube)
        gc.collect()

        if rejection in ('chauvenet', 'rcr'):
            if lo is None:
                lo = True
            if hi is None:
                hi = True
            if rejection == 'rcr':
                mean_type = 'median'
                sigma_type = 'absdev68'
            else:
                mean_type = 'mean'
                sigma_type = 'stddev'
            datacube.mask = chauvenet(
                datacube, min_vals=min_keep, mean=mean_type, sigma=sigma_type,
                clip_lo=lo, clip_hi=hi)
        elif rejection == 'iraf':
            if lo is None:
                lo = 1
            if hi is None:
                hi = 1
            if n - (lo + hi) < min_keep:
                raise ValueError(
                    'IRAF rejection with lo={}, hi={} would keep less than '
                    '{} values for a {}-image set'.format(lo, hi, min_keep, n))
            if lo or hi:
                # Mask "lo" smallest values and "hi" largest values along
                # the 0th axis
                order = datacube.argsort(0)
                mg = tuple(i.ravel() for i in np.indices(datacube.shape[1:]))
                for j in range(-hi, lo):
                    datacube.mask[(order[j].ravel(),) + mg] = True
                del order, mg
                gc.collect()
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
            gc.collect()
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
                    res = np.median(datacube, 0)
            else:
                if isinstance(datacube, ma.MaskedArray):
                    res = ma.masked_array(
                        np.nanpercentile(
                            datacube.filled(np.nan), percentile, 0),
                        np.zeros_like(datacube[0], bool))
                    res[np.isnan(res)] = True
                else:
                    res = np.percentile(datacube, percentile, 0)
        else:
            raise ValueError('Unknown stacking mode "{}"'.format(mode))

        if isinstance(res, ma.MaskedArray) and initial_mask is not None:
            # OR rejection mask with the OR of pre-rejection masks
            res.mask |= initial_mask

        chunks.append(res)

        if callback is not None:
            callback(
                progress +
                ((0.5 if scaling else 0) +
                 min(chunk + chunksize, data_height)/data_height /
                 (2 if scaling else 1))*progress_step)

    if len(chunks) > 1:
        res = ma.vstack(chunks)
    else:
        res = chunks[0]
    if isinstance(res, ma.MaskedArray) and (
            res.mask is None or res.mask is False or not res.mask.any()):
        res = res.data

    if transformations and equalize_global and equalize_order > 0:
        # The background equalization least-squares system is degenerate
        # (the coefficient matrix rank is npar less than N (see above), so
        # in fact we have a family of least-squares solutions; choose one of
        # them that makes the resulting mosaic background as flat as possible
        # by fitting the model to the entire mosaic
        res = global_equalize(res, equalize_order)

    return res, rej_percent


def combine(input_data: List[Union[pyfits.HDUList,
                                   Tuple[Union[np.ndarray, ma.MaskedArray],
                                         pyfits.Header]]],
            mode: str = 'average', scaling: Optional[str] = None,
            rejection: Optional[str] = None, min_keep: int = 2,
            propagate_mask: bool = True, percentile: float = 50.0,
            lo: Optional[Union[bool, int, float]] = None,
            hi: Optional[Union[bool, int, float]] = None,
            equalize_additive: bool = False, equalize_order: int = 0,
            equalize_multiplicative: bool = True,
            equalize_global: bool = False,
            smart_stacking: Optional[str] = None, max_mem_mb: float = 100.0,
            callback: Optional[callable] = None) \
        -> List[Tuple[np.ndarray, pyfits.Header]]:
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
        outliers, "chauvenet" - use classic Chauvenet rejection, "rcr" - use
        super-simplified Robust Chauvenet Rejection, "iraf" - IRAF-like
        clipping of `lo` lowest and `hi` highest values, "minmax" - reject
        values outside the absolute lower and upper limits (use with caution as
        `min_keep` below is not guaranteed, and you may end up in all values
        rejected for some or even all pixels), "sigclip" - iteratively reject
        pixels below and/or above the baseline
    :param min_keep: minimum values to keep during rejection
    :param propagate_mask: when combining masked images, mask the output pixel
        if it's masked in at least one input image
    :param percentile: for `mode`="percentile", default: 50 (median)
    :param lo:
        `rejection` = "iraf": (int) number of lowest values to clip; default: 1
        `rejection` = "minmax": (float) reject values below this limit;
            default: not set
        `rejection` = "sigclip": (float) reject values more than `lo` sigmas
            below the baseline; default: 3
        `rejection` = "chauvenet" or "rcr": (bool) reject negative outliers;
            default: True
    :param hi:
        `rejection` = "iraf": (int) number of highest values to clip;
            default: 1;
        `rejection` = "minmax": (float) reject values above this limit;
            default: not set
        `rejection` = "sigclip": (float) reject values more than `hi` sigmas
            above the baseline; default: 3
        `rejection` = "chauvenet" or "rcr": (bool) reject positive outliers;
            default: True
    :param equalize_additive: enable additive equalization for mosaicing
    :param equalize_order: additive equalization polynomial order
    :param equalize_multiplicative: enable multiplicative mosaic equalization
    :param equalize_global: enable additive background flattening using
        `equalize_order` model
    :param smart_stacking: enable smart stacking: automatically exclude those
        images from the stack that will not improve its quality in a certain
        sense; currently supported modes:
            "SNR": don't include image if it won't improve the resulting
                signal-to-noise ratio of sources; suitable for deep-sky imaging
                to reject images taken through clouds or with bad alignment
            "sharpness": don't include image if it will not improve sharpness
        WARNING. Enabling smart stacking may dramatically increase
                 the processing time.
    :param max_mem_mb: maximum amount of RAM in megabytes to use during
        stacking
    :param callback: optional callable
            def callback(percent: float) -> None
        that is periodically called to update the progress of stacking
        operation

    :return: list of pairs (data, header) of the same length as the number
        of HDUs in the input FITS images (one if a (data, header) list
        was supplied on input), with data set to combined array(s) and
        header(s) copied from one of the input images and modified to reflect
        the stacking mode and parameters
    """
    if len(input_data) < 2:
        raise ValueError('No data to combine')

    if not smart_stacking:
        score_func = None
    else:
        try:
            score_func = smart_stacking_score[smart_stacking]
        except KeyError:
            raise ValueError(
                'Unknown smart stacking mode "{}"'.format(smart_stacking))

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
    total_progress = 0
    progress_step = 100/nhdus
    if score_func is not None:
        progress_step /= len(input_data) + 1
    for hdu_no in range(nhdus):
        # Check image sizes
        data_width = data_height = 0
        for data in input_data:
            if isinstance(data, pyfits.HDUList):
                hdr = data[hdu_no].header
            else:
                hdr = data[1]
            w, h = hdr['NAXIS1'], hdr['NAXIS2']
            if not data_width:
                data_width, data_height = w, h
            elif (data_width, data_height) != (w, h):
                raise ValueError(
                    'Trying to combine arrays with non-matching dimensions: '
                    '{:d}x{:d} and {:d}x{:d}'.format(
                        data_width, data_height, w, h))

        # Stack all input images
        res, rej_percent = _do_combine(
            hdu_no, total_progress, progress_step, data_width, data_height,
            input_data, mode, scaling, rejection, min_keep, propagate_mask,
            percentile, lo, hi, equalize_additive, equalize_order,
            equalize_multiplicative, equalize_global, max_mem_mb, callback)
        total_progress += progress_step

        final_data = list(input_data)
        if score_func is not None and len(final_data) > 1:
            # Smart stacking mode; try excluding images one by one; keep
            # the image if excluding it does not improve the score
            score = score_func(res)
            for image_to_exclude in input_data:
                if len(final_data) < 2:
                    break
                new_data = list(final_data)
                # Cannot just do new_data.remove(image_to_exclude) in older
                # Python versions
                for i, x in enumerate(final_data):
                    if x is image_to_exclude:
                        del new_data[i]
                        break
                new_res, new_rej_percent = _do_combine(
                    hdu_no, total_progress, progress_step, data_width,
                    data_height, new_data, mode, scaling, rejection, min_keep,
                    propagate_mask, percentile, lo, hi, equalize_additive,
                    equalize_order, equalize_multiplicative, equalize_global,
                    max_mem_mb, callback)
                total_progress += progress_step
                new_score = score_func(new_res)
                if new_score > score:
                    # Score improved, reject the current image
                    score, final_data = new_score, new_data
                    res, rej_percent = new_res, new_rej_percent

        # Update FITS headers, start from the first image
        headers = [f[hdu_no].header if isinstance(f, pyfits.HDUList) else f[1]
                   for f in final_data]
        hdr = headers[0].copy(strip=True)

        exp_lengths = [
            h['EXPTIME']
            if 'EXPTIME' in h and not isinstance(h['EXPTIME'], str)
            else h['EXPOSURE'] if 'EXPOSURE' in h and
            not isinstance(h['EXPOSURE'], str) else None
            for h in headers]
        have_exp_lengths = exp_lengths.count(None) < len(exp_lengths)
        if have_exp_lengths:
            exp_lengths = np.array(
                [float(l) if l is not None else 0.0 for l in exp_lengths])
        t_start, t_cen, t_end = tuple(zip(*[get_fits_time(h)
                                            for h in headers]))

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
            hdr['SUMEXP'] = (
                float(exp_lengths.sum()), '[s] Sum of all exposures times')

        try:
            hdr['DATE-OBS'] = (
                min([t for t in t_start if t is not None]).isoformat(),
                'Start time of the first exposure in stack')
        except ValueError:
            # No exposure start times
            pass

        if t_cen.count(None) < len(t_cen):
            # Calculate the average center time by converting to seconds since
            # the first exposure
            known_epochs = (
                np.array([i for i, t in enumerate(t_cen) if t is not None]),)
            t0 = min([t for t in t_cen if t is not None])
            epochs = np.array(
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
                max([t for t in t_end if t is not None]).isoformat(),
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
        hdr['REJPRCNT'] = (float(rej_percent/data_width/data_height /
                                 len(final_data)*100),
                           'Percentage of rejected pixels')

        hdr['SCAMETH'] = (
            scaling.upper() if scaling is not None else 'NONE',
            'Scale method used in combining')

        hdr['WGTMETH'] = ('NONE', 'Weight method used in combining')

        hdr['NCOMB'] = (len(final_data), 'Number of images used in combining')
        hdr['SMARTSTK'] = (str(smart_stacking), 'Smart stacking mode')

        for i, im in enumerate(final_data):
            if isinstance(im, pyfits.HDUList) and im.filename():
                hdr['IMG_{:04d}'.format(i)] = (
                    os.path.basename(im.filename()), 'Component filename')

        output.append((res, hdr))

    return output
