"""
Image stacking.

:func:`~combine()`: combine a series of FITS images using the various stacking
modes with optional scaling and outlier rejection.
"""

import gc
import os.path
import logging
from typing import BinaryIO, Callable, Optional
from datetime import timedelta
from tempfile import TemporaryFile

import numpy as np
from numpy import ma
import astropy.io.fits as pyfits

from ..util.stats import chauvenet
from ..util.fits import get_fits_time
from .mosaicing import get_equalization_transforms, global_equalize
from .smart_stacking import smart_stacking_score
from .util import get_data


__all__ = ['combine']


def _calc_scaling(scaling: str,
                  percentile: float,
                  input_data: list[callable],
                  callback: Optional[Callable] = None,
                  progress: float = 0,
                  progress_step: float = 0) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Calculate scaling factors and offsets

    :param scaling: scaling mode
    :param percentile: percentile value for `scaling` = "percentile"
    :param input_data: list of callables returning 2D image data
    :param callback: optional progress callback
    :param progress: overall progress at the beginning of the current step
    :param progress_step: fraction of overall progress for the current step

    :return: offsets, scaling factors, and their averages
    """
    n = len(input_data)
    offsets = np.zeros(n)
    scaling_factors = np.zeros(n)
    for data_no, f in enumerate(input_data):
        data = f()
        ofs = 0
        if scaling == 'average':
            avg = data.mean()

        elif scaling == 'median' or scaling == 'percentile' and percentile == 50:
            avg = np.median(data) if not isinstance(data, ma.MaskedArray) else ma.median(data)

        elif scaling == 'percentile':
            avg = np.percentile(data, percentile) if not isinstance(data, ma.MaskedArray) \
                else np.percentile(data.compressed(), percentile)

        elif scaling == 'mode':
            # Compute modal values from histograms; convert to integer and assume 2 x 16-bit data range
            if isinstance(data, ma.MaskedArray):
                data = data.compressed()
            else:
                data = data.ravel()
            min_val = data.min(initial=0)
            avg = np.argmax(np.bincount((data - min_val).clip(0, 2*0x10000 - 1).astype(np.int32))) + min_val

        elif scaling == 'histogram':
            # Approximate histogram peak and tail matching: subtract mode, then divide by median
            if isinstance(data, ma.MaskedArray):
                data = data.compressed()
            else:
                data = data.ravel()
            min_val = data.min(initial=0)
            ofs = -np.argmax(np.bincount((data - min_val).clip(0, 2*0x10000 - 1).astype(np.int32))) - min_val
            avg = data.mean() + ofs

        else:
            raise ValueError(f'Unknown scaling mode "{scaling}"')

        if avg <= 0:
            # To make sure that all images are scaled by a positive factor, add a constant offset to the image so that
            # its average = 1
            ofs, avg = 1 - avg, 1

        offsets[data_no], scaling_factors[data_no] = ofs, avg
        del data
        gc.collect()

        if callback is not None:
            callback(progress + (data_no + 1)/n*progress_step)

    # Normalize to the first frame with positive average
    k_ref = scaling_factors[0]
    if k_ref == 1:
        for k in scaling_factors[1:]:
            if k != 1:
                k_ref = k
                break

    # Invert scaling factors
    scaling_factors[scaling_factors != 0] = k_ref/scaling_factors[scaling_factors != 0]

    # Calculate the average offset and scale to restore the final stack counts
    inv_offset = -offsets.mean()
    if scaling_factors.any():
        inv_scaling_factor = (1/scaling_factors[scaling_factors != 0]).mean()
    else:
        inv_scaling_factor = 1

    return offsets, scaling_factors, inv_offset, inv_scaling_factor


def _apply_equalization(equalize_additive: bool, equalize_order: int,
                        equalize_multiplicative: bool,
                        transformations: dict[int, np.ndarray],
                        chunk: int,
                        datacube: list[np.ndarray | ma.MaskedArray]) -> None:
    """
    Apply equalization transformation to chunk of data in place

    :param equalize_additive: enable additive equalization
    :param equalize_order: additive equalization order
    :param equalize_multiplicative: enable multiplicative equalization
    :param transformations: equalization transformations as returned by :func:`get_equalization_transforms`
    :param chunk: data chunk offset
    :param datacube: chunk of 3D datacube
    """
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
            data /= coeffs[pofs]
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


def _combine_data(mode: str,
                  percentile: float,
                  datacube: np.ndarray,
                  initial_mask: np.ndarray | None) -> np.ndarray:
    """
    Stack data using the given mode; used by :func:`_do_combine`

    :param mode: stacking mode
    :param percentile: percentile value for `mode` = "percentile"
    :param datacube: datacube or its chunk to stack
    :param initial_mask: optional initial mask to OR with the result

    :return: combined 2D array
    """
    if mode == 'average':
        res = datacube.mean(0)
    elif mode == 'sum':
        res = datacube.sum(0)
    elif mode == 'median' or mode == 'percentile' and percentile == 50:
        if isinstance(datacube, ma.MaskedArray):
            res = ma.median(datacube, 0)
        else:
            res = np.median(datacube, 0)
    elif mode == 'percentile':
        if isinstance(datacube, ma.MaskedArray):
            res = ma.masked_array(
                np.nanpercentile(datacube.filled(np.nan), percentile, 0),
                np.zeros_like(datacube[0], bool))
            res.mask[np.isnan(res)] = True
        else:
            res = np.percentile(datacube, percentile, 0)
    else:
        raise ValueError('Unknown stacking mode "{}"'.format(mode))

    if isinstance(res, ma.MaskedArray) and initial_mask is not None:
        # OR rejection mask with the OR of pre-rejection masks
        res.mask |= initial_mask

    return res


def _get_data_override_mask(i: int,
                            input_data: list[callable],
                            data_width: int,
                            data_height: int,
                            mask_files: list[BinaryIO],
                            start: int = 0,
                            end: int | None = None,
                            downsample: int = 1) -> ma.MaskedArray:
    """
    Image data retrieval function that supports rejection mask stored in a separate temporary file

    :param i: index of the image in `input_data` and `mask_files`
    :param input_data: list of data retrieval callables returning image data
    :param data_width: initial image width
    :param data_height: initial image height
    :param mask_files: list of open :class:`tempfile.TemporaryFile` objects holding rejection masks for each image
    :param start: first row of chunk
    :param end: last row of chunk
    :param downsample: downsampling factor
    :return:
    """
    # Retrieve downsampled chunk of original data
    data = input_data[i](start=start, end=end, downsample=downsample)

    # Retrieve rejection mask
    mf = mask_files[i]
    if mf is None:
        return data

    mf.seek(0)
    if end is None or end > data_height:
        end = data_height
    h, w = end - start, data_width
    mask = np.fromfile(mf, dtype=np.bool_, offset=start*w, count=h*w).reshape((h, w))

    # Downsample mask if needed
    if downsample >= 2:
        width = w//downsample
        height = h//downsample
        if h/downsample % 1:
            mask = mask[:height*downsample]
        if w/downsample % 1:
            mask = mask[:, :width*downsample]
        mask = (mask.reshape(height, downsample, width, downsample)
                .sum(3).sum(1)/downsample**2).astype(np.bool_)

    # Replace the original image mask with rejection mask
    if isinstance(data, ma.MaskedArray):
        data.mask = mask
    else:
        data = ma.masked_array(data, mask)

    return data


def _do_combine(input_data: list[callable],
                data_width: int,
                data_height: int,
                mode: str = 'average',
                percentile: float = 50.0,
                scaling: str | None = None,
                scaling_percentile: float = 50.0,
                prescaling: str | None = None,
                prescaling_percentile: float = 50.0,
                rejection: str | None = None,
                min_keep: int = 2,
                lo: bool | int | float | None = None,
                hi: bool | int | float | None = None,
                propagate_mask: bool = True,
                equalize_additive: bool = False,
                equalize_order: int = 1,
                equalize_multiplicative: bool = False,
                multiplicative_percentile: float = 99.9,
                equalize_global: bool = False,
                max_mem_mb: float = 100.0,
                callback: Optional[Callable] = None,
                progress: float = 0,
                progress_step: float = 0,
                save_masks: bool = False) \
        -> tuple[np.ndarray | ma.MaskedArray, float, list[BinaryIO]]:
    """
    Combine the given HDUs from all input images; used by :func:`combine` to get a stack of either all input images or,
    if smart stacking is enabled, of their subset

    :return: image stack data, rejection percent, and list of temporary files holding rejection masks if
        rejection = True and either scaling != None or save_masks = True
    """
    if prescaling and not rejection:
        # Disable prescaling if rejection was not enabled
        prescaling = None

    if scaling and rejection:
        save_masks = True

    n = len(input_data)
    progress_step /= (
        # Main stacking loop, always present
        1 +
        # Calculate equalization transformations
        int(equalize_additive or equalize_multiplicative) +
        # Compute prescaling parameters
        int(bool(prescaling)) +
        # Compute scaling parameters
        int(bool(scaling)) +
        # If rejection and scaling are enabled, we have one more stacking loop
        # and one more equalization if enabled
        int(bool(scaling) and bool(rejection)) *
        (1 + int(equalize_additive or equalize_multiplicative))
    )

    if equalize_additive or equalize_multiplicative:
        # Equalize the common image parts; suitable for mosaicing
        transformations = get_equalization_transforms(
            progress, progress_step, data_width, data_height, input_data, equalize_additive, equalize_order,
            equalize_multiplicative, multiplicative_percentile, max_mem_mb, callback)
        progress += progress_step
    else:
        transformations = {}

    # Calculate prescaling offsets and factors
    if prescaling:
        prescaling_offsets, prescaling_factors = _calc_scaling(
            prescaling, prescaling_percentile, input_data, callback, progress, progress_step)[:2]
        progress += progress_step
    else:
        prescaling_offsets, prescaling_factors = [], []

    # Calculate scaling offsets and factors if we won't do rejection; otherwise, do this after rejection
    if scaling and not rejection:
        scaling_offsets, scaling_factors, inv_offset, inv_scaling_factor = _calc_scaling(
            scaling, scaling_percentile, input_data, callback, progress, progress_step)
        progress += progress_step
    else:
        scaling_offsets, scaling_factors = [], []
        inv_offset, inv_scaling_factor = 0, 1

    # Process data in chunks to fit in the maximum amount of RAM allowed
    bytes_per_pixel = max(f(start=0, end=1).itemsize for f in input_data) + 1
    rowsize = total_rowsize = data_width*bytes_per_pixel*n
    if prescaling:
        total_rowsize += rowsize
    if rejection:
        total_rowsize += rowsize
    chunksize = min(max(int(max_mem_mb*(1 << 20)/total_rowsize), 1), data_height)
    while chunksize > 1:
        # Use as small chunks as possible but keep their total number
        if len(list(range(0, data_height, chunksize - 1))) > len(list(range(0, data_height, chunksize))):
            break
        chunksize -= 1
    chunks = []
    if save_masks:
        # Prepare temporary files that will store masks after rejection
        mask_files = [TemporaryFile() for _ in range(n)]
    else:
        mask_files = [None]*n
    rej_percent = 0
    for chunk in range(0, data_height, chunksize):
        unscaled_datacube = [f(start=chunk, end=chunk + chunksize) for f in input_data]

        # Scale data
        if prescaling:
            datacube = []
            for unscaled_data, ofsi, ki in zip(unscaled_datacube, prescaling_offsets, prescaling_factors):
                data = unscaled_data.copy()
                if ofsi:
                    data += ofsi
                if ki not in (0, 1):
                    data *= ki
                datacube.append(data)
        else:
            datacube = unscaled_datacube

        # Equalize backgrounds
        if transformations:
            _apply_equalization(
                equalize_additive, equalize_order, equalize_multiplicative, transformations, chunk, datacube)
            gc.collect()

        initial_mask = None
        if rejection or any(isinstance(data, ma.MaskedArray) for data in datacube):
            datacube = ma.masked_array(datacube, fill_value=np.nan)
            if not datacube.mask.shape:
                # No initially masked data, but we'll need an array instead of mask=False to do slicing operations
                datacube.mask = np.full(datacube.shape, datacube.mask)

            if propagate_mask and not (scaling and rejection):
                # After stacking (and possibly rejection), we'll mask all pixels that are initially masked in at least
                # one image (e.g. edges/corners after alignment); this will be done during a separate stacking step if
                # both rejection and scaling are enabled
                initial_mask = np.logical_or.reduce(datacube.mask, axis=0)
                if not initial_mask.any():
                    initial_mask = None
        else:
            datacube = np.array(datacube)
        gc.collect()

        if rejection in ('chauvenet', 'rcr'):
            if datacube.dtype.name == 'float32':
                # Numba is slower for 32-bit floating point
                datacube = datacube.astype(np.float64)
            elif not datacube.dtype.isnative:
                # Non-native byte order is not supported by Numba
                datacube = datacube.astype(datacube.dtype.newbyteorder())
            chauvenet(
                datacube.data, datacube.mask, min_vals=min_keep,
                mean_type=1 if rejection == 'rcr' else 0,
                sigma_type=1 if rejection == 'rcr' else 0,
                clip_lo=lo, clip_hi=hi)
        elif rejection == 'iraf':
            if lo or hi:
                # Mask "lo" smallest values and "hi" largest values along the 0th axis
                order = datacube.argsort(0)
                mg = tuple(i.ravel() for i in np.indices(datacube.shape[1:]))
                for j in range(-hi, lo):
                    datacube.mask[(order[j].ravel(),) + mg] = True
                del order, mg
                gc.collect()
        elif rejection == 'minmax':
            datacube.mask[(datacube < lo) | (datacube > hi)] = True
            if datacube.mask.all(0).any():
                logging.warning('%d completely masked pixels left after minmax rejection', datacube.mask.all(0).sum())
        elif rejection == 'sigclip':
            max_rej = n - min_keep
            while True:
                avg = datacube.mean(0)
                sigma = datacube.std(0)
                resid = datacube - avg
                outliers = (datacube.mask.sum(0) < max_rej) & (sigma > 0) & ((resid < -lo*sigma) | (resid > hi*sigma))
                if not outliers.any():
                    del avg, sigma, resid, outliers
                    break
                datacube.mask[outliers.nonzero()] = True
            gc.collect()

        if save_masks:
            for data, f in zip(datacube, mask_files):
                data.mask.tofile(f)

        if scaling and rejection:
            # With both scaling and rejection enabled, we cannot stack right away since we'll need the whole image
            # rejection mask after all mask chunks have been saved
            rej_percent += datacube.mask.sum()
        else:
            # Restore original data if scaling was enabled, keep the mask
            if prescaling:
                # With rejection enabled, datacube is always a masked array
                for i in range(n):
                    if isinstance(unscaled_datacube[i], ma.MaskedArray):
                        datacube.data[i] = unscaled_datacube[i].data
                    else:
                        datacube.data[i] = unscaled_datacube[i]

            # Apply scaling
            if scaling:
                for data, ofsi, ki in zip(datacube, scaling_offsets, scaling_factors):
                    if ofsi:
                        data += ofsi
                    if ki not in (0, 1):
                        data *= ki

            if isinstance(datacube, ma.MaskedArray):
                if datacube.mask is None or not datacube.mask.any():
                    # Nothing was rejected
                    datacube = datacube.data
                else:
                    # Calculate the percentage of rejected pixels
                    rej_percent += datacube.mask.sum()

            # Combine data
            chunks.append(_combine_data(mode, percentile, datacube, initial_mask))

        if callback is not None:
            callback(progress + min(chunk + chunksize, data_height)/data_height*progress_step)
    progress += progress_step

    if scaling and rejection:
        # Restore the rejection mask and calculate offsets and scaling factors
        scaling_offsets, scaling_factors, inv_offset, inv_scaling_factor = _calc_scaling(
            scaling, scaling_percentile,
            [lambda i=_i_: _get_data_override_mask(i, input_data, data_width, data_height, mask_files)
             for _i_ in range(n)], callback, progress, progress_step)
        progress += progress_step

        if equalize_additive or equalize_multiplicative:
            # Recompute equalization transformations after rejection
            transformations = get_equalization_transforms(
                progress, progress_step, data_width, data_height,
                [lambda *args, i=_i_, **kwargs: _get_data_override_mask(
                    i, input_data, data_width, data_height, mask_files, *args, **kwargs)
                 for _i_ in range(n)],
                equalize_additive, equalize_order, equalize_multiplicative, multiplicative_percentile, max_mem_mb,
                callback)
            progress += progress_step

        # Scale and stack by chunk
        for chunk in range(0, data_height, chunksize):
            # Load image data with the original mask
            datacube = ma.masked_array([f(start=chunk, end=chunk + chunksize) for f in input_data], fill_value=np.nan)
            if not datacube.mask.shape:
                datacube.mask = np.full(datacube.shape, datacube.mask)
            if propagate_mask:
                initial_mask = np.logical_or.reduce(datacube.mask, axis=0)
                if not initial_mask.any():
                    initial_mask = None
            else:
                initial_mask = None

            # Set mask to that computed by rejection
            for data_no in range(n):
                mf = mask_files[data_no]
                mf.seek(0)
                count = min(chunksize, data_height - chunk)
                datacube.mask[data_no] = np.fromfile(
                    mf, dtype=np.bool_, count=count*data_width,
                    offset=chunk*data_width).reshape((count, data_width))
            gc.collect()

            # Scale data
            for data, ofsi, ki in zip(datacube, scaling_offsets, scaling_factors):
                if ofsi:
                    data += ofsi
                if ki not in (0, 1):
                    data *= ki

            # Equalize backgrounds
            if transformations:
                _apply_equalization(
                    equalize_additive, equalize_order, equalize_multiplicative, transformations, chunk, datacube)
                gc.collect()

            if datacube.mask is None or not datacube.mask.any():
                # Nothing was rejected
                datacube = datacube.data

            # Combine data
            chunks.append(_combine_data(mode, percentile, datacube, initial_mask))

            if callback is not None:
                callback(progress + min(chunk + chunksize, data_height)/data_height*progress_step)
        progress += progress_step

    if len(chunks) > 1:
        res = ma.vstack(chunks)
    else:
        res = chunks[0]
    if isinstance(res, ma.MaskedArray) and (not res.mask.shape or not res.mask.any()):
        res = res.data

    if transformations and equalize_global and equalize_order > 0:
        # The background equalization least-squares system is degenerate (the coefficient matrix rank is npar less than
        # N (see above), so in fact we have a family of least-squares solutions; choose one of them that makes
        # the resulting mosaic background as flat as possible by fitting the model to the entire mosaic
        res = global_equalize(res, equalize_order)

    # Rescale the stack back to the original counts to keep photometry errors correct
    if inv_scaling_factor not in (0, 1):
        res *= inv_scaling_factor
    if inv_offset:
        res += inv_offset

    return res, rej_percent, mask_files


def combine(input_data: list[pyfits.HDUList | tuple[np.ndarray | ma.MaskedArray, pyfits.Header]],
            mode: str = 'average',
            percentile: float = 50.0,
            scaling: str | None = None,
            scaling_percentile: float = 50.0,
            prescaling: str | None = None,
            prescaling_percentile: float = 50.0,
            rejection: str | None = None,
            min_keep: int = 2,
            lo: bool | int | float | None = None,
            hi: bool | int | float | None = None,
            propagate_mask: bool = True,
            equalize_additive: bool = False,
            equalize_order: int = 0,
            equalize_multiplicative: bool = False,
            multiplicative_percentile: float = 99.9,
            equalize_global: bool = False,
            smart_stacking: str | None = None,
            max_mem_mb: float = 100.0,
            callback: Optional[Callable] = None,
            return_headers: bool = True) \
        -> list[tuple[np.ndarray, pyfits.Header] | np.ndarray]:
    """
    Combine a series of FITS images using the various stacking modes with optional scaling and outlier rejection

    :param input_data: two or more FITS images or pairs (data, header) to combine; FITS files must be opened in readonly
        mode and have all the same number of HDUs and, separately for each HDU, the same data dimensions
    :param mode: stacking mode: "average" (default), "sum", "median", or "percentile"
    :param percentile: for `mode`="percentile", default: 50 (median)
    :param scaling: data scaling mode (applied before stacking, normalizes output); possible values::
            None (default): do not scale data;
            "average": scale data to match average values;
            "median": scale to match median;
            "percentile": match the given percentile (median for `scaling_percentile` = 50);
            "mode": match modal values;
            "histogram": histogram peak and tail normalization
    :param scaling_percentile: percentile value for `scaling` = "percentile"
    :param prescaling: pre-rejection scaling mode (applied before rejection, does not normalize output, ignored if
        `rejection` = None); possible values: same as for `scaling`
    :param prescaling_percentile: percentile value for `prescaling` = "percentile"
    :param rejection: outlier rejection mode; possible values::
        None (default): do not reject outliers;
        "chauvenet": use classic Chauvenet rejection;
        "rcr": use super-simplified Robust Chauvenet Rejection;
        "iraf": IRAF-like clipping of `lo` lowest and `hi` highest values;
        "minmax": reject values outside the absolute lower and upper limits (use with caution as `min_keep` below is not
            guaranteed, and you may end up in all values rejected for some or even all pixels);
        "sigclip": iteratively reject pixels below and/or above the baseline
    :param lo:
        `rejection` = "iraf": (int) number of lowest values to clip; default: 1
        `rejection` = "minmax": (float) reject values below this limit; default: not set
        `rejection` = "sigclip": (float) reject values more than `lo` sigmas below the baseline; default: 3
        `rejection` = "chauvenet" or "rcr": (bool) reject negative outliers; default: True
    :param hi:
        `rejection` = "iraf": (int) number of highest values to clip; default: 1;
        `rejection` = "minmax": (float) reject values above this limit; default: not set
        `rejection` = "sigclip": (float) reject values more than `hi` sigmas above the baseline; default: 3
        `rejection` = "chauvenet" or "rcr": (bool) reject positive outliers; default: True
    :param min_keep: minimum values to keep during rejection
    :param propagate_mask: when combining masked images, mask the output pixel if it's masked in at least one input
        image
    :param equalize_additive: enable additive equalization for mosaicing
    :param equalize_order: additive equalization polynomial order
    :param equalize_multiplicative: enable multiplicative mosaic equalization
    :param multiplicative_percentile: calculate equalization scaling factors by comparing pixels at this percentile
    :param equalize_global: enable additive background flattening using `equalize_order` model
    :param smart_stacking: enable smart stacking: automatically exclude those images from the stack that will not
        improve its quality in a certain sense; currently supported modes:
            "SNR": don't include image if it won't improve the resulting signal-to-noise ratio of sources; suitable for
                deep-sky imaging to reject images taken through clouds or with bad alignment
            "sharpness": don't include image if it will not improve sharpness
        WARNING. Enabling smart stacking may dramatically increase the processing time.
    :param max_mem_mb: maximum amount of RAM in megabytes to use during stacking
    :param callback: optional callable
            def callback(percent: float) -> None
        that is periodically called to update the progress of stacking operation
    :param return_headers: for each input HDU#, compile a header from input headers with info about stacking

    :return: list of pairs (data, header) of the same length as the number of HDUs in the input FITS images (one if
        a (data, header) list was supplied on input), with data set to combined array(s) and header(s) copied from one
        of the input images and modified to reflect the stacking mode and parameters; if `return_headers`=False, only
        data is returned
    """
    if prescaling and not rejection:
        # Disable prescaling if rejection was not enabled
        prescaling = None

    n = len(input_data)
    if n < 2:
        raise ValueError('No data to combine')

    # Assign defaults to rejection parameters
    if rejection in ('chauvenet', 'rcr'):
        if lo is None:
            lo = True
        if hi is None:
            hi = True
    elif rejection == 'iraf':
        if lo is None:
            lo = 1
        if hi is None:
            hi = 1
        if n - (lo + hi) < min_keep:
            raise ValueError(
                f'IRAF rejection with lo={lo}, hi={hi} would keep less than {min_keep} values for a {n}-image set')
    elif rejection == 'minmax':
        if lo is None or hi is None:
            raise ValueError('Minmax rejection requires explicit lower and upper limits')
        if lo > hi:
            raise ValueError(f'Lower limit = {lo} > upper limit = {hi} for minmax rejection')
    elif rejection == 'sigclip':
        if lo is None:
            lo = 3
        if hi is None:
            hi = 3
        if lo < 0 or hi < 0:
            raise ValueError(f'Lower and upper limits for sigma clipping must be positive, got lo={lo}, hi={hi}')
    elif rejection:
        raise ValueError(f'Unknown rejection mode "{rejection}"')

    if not smart_stacking:
        score_func = None
    else:
        try:
            score_func = smart_stacking_score[smart_stacking]
        except KeyError:
            raise ValueError(f'Unknown smart stacking mode "{smart_stacking}"')

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
        progress_step /= len(input_data) + 1 + int(bool(rejection))
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
                    f'{data_width:d}x{data_height:d} and {w:d}x{h:d}')

        # Stack all input images; obtain rejection masks, which will be needed in smart stacking mode
        res, rej_percent, mask_files = _do_combine(
            [lambda *args, i=_i_, **kwargs: get_data(input_data[i], hdu_no, *args, **kwargs)
             for _i_ in range(n)],
            data_width, data_height, mode, percentile=percentile, scaling=scaling,
            scaling_percentile=scaling_percentile, prescaling=prescaling, prescaling_percentile=prescaling_percentile,
            rejection=rejection, lo=lo, hi=hi, min_keep=min_keep, propagate_mask=propagate_mask,
            equalize_additive=equalize_additive, equalize_order=equalize_order,
            equalize_multiplicative=equalize_multiplicative, multiplicative_percentile=multiplicative_percentile,
            equalize_global=equalize_global, max_mem_mb=max_mem_mb, callback=callback, progress=total_progress,
            progress_step=progress_step, save_masks=rejection and score_func is not None and len(input_data) > 1)
        total_progress += progress_step

        final_data = list(input_data)
        if score_func is not None and len(final_data) > 1:
            # Smart stacking mode; try excluding images one by one; keep the image if excluding it does not improve
            # the score
            score = score_func(res)
            for image_to_exclude in input_data:
                if len(final_data) < 2:
                    break
                new_data = list(final_data)
                new_mask_files = list(mask_files)
                # Cannot just do new_data.remove(image_to_exclude) in older Python versions
                for i, x in enumerate(final_data):
                    if x is image_to_exclude:
                        del new_data[i], new_mask_files[i]
                        break
                # For the sake of speed, disable rejection and equalization during smart stacking; use the previously
                # computed rejection masks stored in mask_files
                new_data_getters = [
                    lambda *args, i=_i_, **kwargs: get_data(new_data[i], hdu_no, *args, **kwargs)
                    for _i_ in range(len(new_data))]
                if new_mask_files.count(None) == len(new_mask_files):
                    # No rejection masks stored in intermediate files, use normal data getters
                    new_input_data = new_data_getters
                else:
                    # Use data getters overriding mask with the one stored in
                    # intermediate file
                    new_input_data = [
                        lambda *args, i=_i_, **kwargs: _get_data_override_mask(
                            i, new_data_getters, data_width, data_height, new_mask_files, *args, **kwargs)
                        for _i_ in range(len(new_data))]
                new_res, new_rej_percent, _ = _do_combine(
                    new_input_data, data_width, data_height, mode, percentile=percentile, scaling=scaling,
                    scaling_percentile=scaling_percentile, prescaling=prescaling,
                    prescaling_percentile=prescaling_percentile, rejection=None, lo=lo, hi=hi, min_keep=min_keep,
                    propagate_mask=propagate_mask, equalize_additive=False, equalize_multiplicative=False,
                    equalize_global=False, max_mem_mb=max_mem_mb, callback=callback, progress=total_progress,
                    progress_step=progress_step)
                del new_input_data, new_data_getters
                total_progress += progress_step
                new_score = score_func(new_res)
                if new_score > score:
                    # Score improved, reject the current image
                    score, final_data = new_score, new_data
                    res, rej_percent = new_res, new_rej_percent
                    mask_files = new_mask_files
                del new_data, new_mask_files, new_res
                gc.collect()

            # Re-stack the final set of images with all initially requested features enabled
            if rejection or equalize_additive or equalize_multiplicative or equalize_global:
                res, rej_percent, _ = _do_combine(
                    [lambda *args, i=_i_, **kwargs: get_data(final_data[i], hdu_no, *args, **kwargs)
                     for _i_ in range(len(final_data))], data_width, data_height,
                    mode, percentile=percentile, scaling=scaling, scaling_percentile=scaling_percentile,
                    prescaling=prescaling, prescaling_percentile=prescaling_percentile, rejection=rejection, lo=lo,
                    hi=hi, min_keep=min_keep, propagate_mask=propagate_mask, equalize_additive=equalize_additive,
                    equalize_order=equalize_order, equalize_multiplicative=equalize_multiplicative,
                    multiplicative_percentile=multiplicative_percentile, equalize_global=equalize_global,
                    max_mem_mb=max_mem_mb, callback=callback, progress=total_progress, progress_step=progress_step)
                total_progress += progress_step

        del mask_files
        gc.collect()

        if not return_headers:
            output.append(res)
            continue

        # Update FITS headers, start from the first image
        headers: list[pyfits.Header] = [f[hdu_no].header if isinstance(f, pyfits.HDUList) else f[1] for f in final_data]
        hdr = headers[0].copy(strip=True)

        exp_lengths = [
            h['EXPTIME'] if 'EXPTIME' in h and not isinstance(h['EXPTIME'], str)
            else h['EXPOSURE'] if 'EXPOSURE' in h and not isinstance(h['EXPOSURE'], str)
            else None
            for h in headers]
        have_exp_lengths = exp_lengths.count(None) < len(exp_lengths)
        if have_exp_lengths:
            exp_lengths = np.array([float(l) if l is not None else 0.0 for l in exp_lengths])
        t_start, t_cen, t_end = tuple(zip(*[get_fits_time(h) for h in headers]))

        hdr['FILTER'] = (','.join({h['FILTER'] for h in headers if 'FILTER' in h}), 'Filter(s) used when taking images')

        hdr['OBSERVAT'] = (','.join({h['OBSERVAT'] for h in headers if 'OBSERVAT' in h}),
                           'Observatory or telescope name(s)')

        if have_exp_lengths:
            hdr['EXPTIME'] = hdr['EXPOSURE'] = (
                float(exp_lengths.sum() if mode == 'sum' else exp_lengths.mean()), '[s] Effective exposure time')
            hdr['SUMEXP'] = (float(exp_lengths.sum()), '[s] Sum of all exposures times')

        try:
            hdr['DATE-OBS'] = (
                min([t for t in t_start if t is not None]).isoformat(), 'Start time of the first exposure in stack')
        except ValueError:
            # No exposure start times
            pass

        if t_cen.count(None) < len(t_cen):
            # Calculate the average center time by converting to seconds since the first exposure
            known_epochs = (np.array([i for i, t in enumerate(t_cen) if t is not None]),)
            t0 = min([t for t in t_cen if t is not None])
            epochs = np.array([(t - t0).total_seconds() for t in t_cen if t is not None])
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
            hdr['DATE-CEN'] = ((t0 + timedelta(seconds=tc)).isoformat(), 'Weighted central time of image stack')

        try:
            hdr['DATE-END'] = (
                max([t for t in t_end if t is not None]).isoformat(), 'Stop time of the last exposure in stack')
        except ValueError:
            # No exposure stop times
            pass

        gains = []
        for h in headers:
            try:
                gains.append(float(h['GAIN']))
            except (KeyError, ValueError):
                pass
        if gains:
            gain = np.mean(gains)
            # See IRAF/DAOPHOT manual, p.20
            if mode == 'median' or mode == 'percentile' and abs(percentile - 50) < 1e-7:
                gain *= len(headers)*2/3
            elif mode != 'sum':
                gain *= len(headers)
            hdr['GAIN'] = (gain, '[e-/count] Effective gain of the stack')

        hdr['COMBMETH'] = (mode.upper(), 'Stacking method')
        if mode == 'percentile':
            hdr['PERCNTLE'] = (percentile, 'Stacking percentile value')

        hdr['REJMETH'] = (rejection.upper() if rejection is not None else 'NONE', 'Rejection method used in combining')
        if rejection in ('chauvenet', 'rcr'):
            hdr['REJLOW'] = (bool(lo), 'Reject negative outliers')
            hdr['REJHIGH'] = (bool(hi), 'Reject positive outliers')
        elif rejection == 'iraf':
            hdr['REJLOW'] = (int(lo), 'Number of low pixels rejected')
            hdr['REJHIGH'] = (int(hi), 'Number of high pixels rejected')
        elif rejection == 'minmax':
            hdr['REJLOW'] = (lo, 'Lower rejection threshold')
            hdr['REJHIGH'] = (hi, 'Upper rejection threshold')
        elif rejection == 'sigclip':
            hdr['REJLOW'] = (float(lo), 'Lower sigma used with rejection')
            hdr['REJHIGH'] = (float(hi), 'Upper sigma used with rejection')
        hdr['REJPRCNT'] = (float(rej_percent/data_width/data_height/len(final_data)*100),
                           'Percentage of rejected pixels')

        hdr['PRESCAL'] = (prescaling.upper() if prescaling is not None else 'NONE', 'Prescaling method')
        hdr['SCALMETH'] = (scaling.upper() if scaling is not None else 'NONE', 'Scaling method')

        hdr['WGTMETH'] = ('NONE', 'Weight method used in combining')

        hdr['NCOMB'] = (len(final_data), 'Number of images used in combining')
        hdr['SMARTSTK'] = (str(smart_stacking), 'Smart stacking mode')

        # Save component names; use either Afterglow Workbench filename if available or FITS filename otherwise
        for i, im in enumerate(final_data):
            if isinstance(im, pyfits.HDUList):
                try:
                    name = im[hdu_no].header['AGFILNAM']
                except KeyError:
                    name = os.path.basename(im.filename())
            else:
                try:
                    name = im[1]['AGFILNAM']
                except KeyError:
                    name = None
            if name:
                hdr['IMG_{:04d}'.format(i)] = (name, 'Component filename')

        output.append((res, hdr))

    return output
