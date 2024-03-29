"""
Math behind image mosaicing
"""

import gc
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from numpy import ma
from numpy.linalg import lstsq
from scipy.sparse import csc_array
from scipy.sparse.linalg import lsqr as sparse_lstsq


__all__ = ['get_equalization_transforms', 'global_equalize']


def _get_overlap(data: Union[np.ndarray, ma.MaskedArray],
                 other_data: Union[np.ndarray, ma.MaskedArray]) \
        -> np.ndarray:
    """
    Return the overlap of two equally-shaped optionally masked images

    :param data: first image data array
    :param other_data: second image data array

    :return: boolean array of the same shape as inputs; True means
        an overlapping pixel = the one that is not masked in both images;
        special case: scalar True means 100% overlap, returned if both images
        are regular (non-masked) arrays
    """
    if isinstance(data, ma.MaskedArray):
        if isinstance(other_data, ma.MaskedArray):
            return (~data.mask) & (~other_data.mask)
        return ~data.mask
    if isinstance(other_data, ma.MaskedArray):
        return ~other_data.mask
    return np.array(True)


def get_equalization_transforms(
        progress: float, progress_step: float, data_width: int,
        data_height: int, input_data: List[callable],
        equalize_additive: bool = False, equalize_order: int = 0,
        equalize_multiplicative: bool = True,
        multiplicative_percentile: float = 99.9, max_mem_mb: float = 100.0,
        callback: Optional[Callable] = None) -> Dict[int, np.ndarray]:
    """
    Calculate tile equalization transformations that make the individual tile
    counts match in the regions of overlap

    :param progress: total stacking job progress by the equalization stage
    :param progress_step: percentage of total stacking job progress for each
        individual stacking
    :param data_width: width of images being stacked
    :param data_height: height of images being stacked
    :param input_data: list of callables
            get_data(start: int = 0, end: Optional[int] = None,
                     downsample: int = 1) -> Union[np.ndarray, ma.MaskedArray)
        returning each data array on the fly
    :param equalize_additive: enable additive equalization
    :param equalize_order: additive equalization polynomial order
    :param equalize_multiplicative: enable multiplicative equalization
    :param multiplicative_percentile: calculate equalization scaling factors
        by comparing pixels at this percentile
    :param max_mem_mb: approximate maximum amount of RAM in megabytes that
        the algorithm is allowed to use
    :param callback: optional progress update callable with signature
        def callback(percent: float) -> None

    :return: transformations[image index] = array of parameters p, with p[i]
        being the coefficient of the equalization model in the following order:
        1, x, y, x^2, xy, y^2, etc., depending on `equalize_order`
    """
    n = len(input_data)
    bytes_per_pixel = []
    for f in input_data:
        bytes_per_pixel.append(f(start=0, end=1).itemsize)
        gc.collect()
    bytes_per_pixel = max(bytes_per_pixel) + 1
    transformations, overlaps, images_with_overlaps = {}, {}, []
    progress_step /= 2*(2 + int(equalize_additive) +
                        int(equalize_multiplicative))

    # Step 1: count the total number of overlapping pixels
    m, skip = 0, []
    for data_no, f in enumerate(input_data):
        data = f()
        for other_data_no, other_f in enumerate(input_data):
            if other_data_no == data_no or \
                    (data_no, other_data_no) in skip:
                continue
            skip.append((other_data_no, data_no))
            overlap = _get_overlap(data, other_f())
            if overlap.shape:
                m += overlap.sum()
            else:
                m += data_width*data_height  # full overlap
            del overlap
        del data
        gc.collect()
        if callback is not None:
            callback(progress + (data_no + 1)/n*progress_step)
    progress += progress_step
    downsample = int(np.ceil(
        np.sqrt(m*(8 + 2*bytes_per_pixel)/max_mem_mb/(1 << 20))))

    # Step 2: temporarily downsample input images so that the overlapping
    # pixel data fits in the allowed amount of RAM and store overlaps for
    # each image
    m, skip = 0, []
    for data_no, f in enumerate(input_data):
        data = f(downsample=downsample)
        # Identify all available overlaps with the other images
        overlaps_for_file = {}
        for other_data_no, other_f in enumerate(input_data):
            if other_data_no == data_no or \
                    (data_no, other_data_no) in skip:
                continue

            # Count each overlap only once
            skip.append((other_data_no, data_no))

            gc.collect()
            other_data = other_f(downsample=downsample)
            overlap = _get_overlap(data, other_data)
            if not overlap.any():
                continue

            # Found an overlap; save its coordinates and original
            # pixel values for both intersecting images
            if overlap.shape:
                overlap_y, overlap_x = overlap.nonzero()
                overlap_data1 = data[overlap]
                if isinstance(overlap_data1, ma.MaskedArray):
                    overlap_data1 = overlap_data1.data
                overlap_data2 = other_data[overlap]
                if isinstance(overlap_data2, ma.MaskedArray):
                    overlap_data2 = overlap_data2.data
                m += overlap.sum()
            else:
                overlap_y, overlap_x = np.indices(data.shape)
                if isinstance(data, ma.MaskedArray):
                    overlap_data1 = data.data
                else:
                    overlap_data1 = data
                if isinstance(other_data, ma.MaskedArray):
                    overlap_data2 = other_data.data
                else:
                    overlap_data2 = other_data
                m += data.shape[0]*data.shape[1]
            del overlap, other_data
            if downsample > 1:
                overlap_x *= downsample
                overlap_y *= downsample
            overlaps_for_file[other_data_no] = (
                overlap_x.astype(np.int32).ravel(),
                overlap_y.astype(np.int32).ravel(),
                overlap_data1.ravel(), overlap_data2.ravel())
            del overlap_data1, overlap_data2, overlap_x, overlap_y
            if data_no not in images_with_overlaps:
                images_with_overlaps.append(data_no)
            if other_data_no not in images_with_overlaps:
                images_with_overlaps.append(other_data_no)

        if overlaps_for_file:
            overlaps[data_no] = overlaps_for_file
            del overlaps_for_file

        del data

        gc.collect()
        if callback is not None:
            callback(progress + (data_no + 1)/n*progress_step)
    progress += progress_step

    if overlaps:
        if equalize_multiplicative:
            # Step 3: find the multiplicative pixel value transformations;
            # similar to additive with one model parameter: I = cs, which leads
            # to least-squares system c_i I_j - c_j I_i = 0 (i != 0),
            # c_j = I_j/I_0, c_0 = 1 for N parameters c_i
            nn = len(images_with_overlaps) - 1
            # noinspection PyUnboundLocalVariable
            mm = sum(len(overlaps_for_file)
                     for overlaps_for_file in overlaps.values())
            param_offset = {}
            ofs = 0
            for i in images_with_overlaps:
                if i:
                    param_offset[i] = ofs
                    ofs += 1

            # Build the least-squares matrices
            a_lsq = np.zeros((mm, nn))
            b_lsq = np.zeros(mm)
            row = 0
            # noinspection PyUnboundLocalVariable
            for i, overlaps_for_file in overlaps.items():
                for j, (_, _, d1, d2) in overlaps_for_file.items():
                    if i:
                        a_lsq[row, param_offset[i]] = np.percentile(
                            d2, multiplicative_percentile)
                        a_lsq[row, param_offset[j]] = -np.percentile(
                            d1, multiplicative_percentile)
                    else:
                        a_lsq[row, param_offset[j]] = np.percentile(
                            d1, multiplicative_percentile)
                        b_lsq[row] = np.percentile(
                            d2, multiplicative_percentile)
                    row += 1

                if callback is not None:
                    callback(progress + (i + 1)/len(overlaps)*progress_step)
            progress += progress_step

            if not equalize_additive:
                del overlaps
            gc.collect()

            # Compute the least-squares solution
            params = lstsq(a_lsq, b_lsq, rcond=None)[0]
            for i, ofs in param_offset.items():
                transformations[i] = params[ofs:ofs+1]

        if equalize_additive:
            # Step 4: find the additive pixel value transformations for each
            # image that minimize the difference in the overlapping areas.
            # The model is I = p0 + p1*x + p2*y + p3*x^2 + p4*xy + p5*y^2 + s
            # (s is the true signal). As long as the transformations are linear
            # with respect to its parameters, this leads to a sparse linear
            # least-squares problem AX = B with MxN coefficient matrix A,
            # M-vector of observed differences B, and N-vector of parameters X;
            # N = (number of model parameters) x (number of images that have at
            # least one overlap), M = (total number of overlapping points) =
            # sum(M_k), where M_k is the number of points in k-th overlap,
            # k = 1...K, K = (total number of overlaps).
            npar = (equalize_order + 1)*(equalize_order + 2)//2
            n = npar*len(images_with_overlaps)

            # Since not all input images may have overlaps, the offset of
            # each set of npar parameters in the N-element vector of parameters
            # X for i-th image is not simply 3 x npar; build a mapping to
            # easily calculate this offset
            param_offset = {}
            ofs = 0
            for i in images_with_overlaps:
                param_offset[i] = ofs
                ofs += npar

            # Build the least-squares matrices
            use_sparse = m*n*8 > max_mem_mb*(1 << 20)
            a_gen = {}  # only used if sparse
            if use_sparse:
                # Too much RAM for a regular array, use scipy.sparse;
                # a_data[col] is a list of (row#, data) pairs representing
                # vertical slices
                a_lsq = None
            else:
                a_lsq = np.zeros((m, n))
            b_lsq = np.empty(m, float)
            row = 0
            # noinspection PyUnboundLocalVariable
            for i, overlaps_for_file in overlaps.items():
                for j, (x, y, d1, d2) in overlaps_for_file.items():
                    npoints = len(x)

                    # Compute all required powers of x and y
                    x_pow, y_pow = [1], [1]
                    if equalize_order > 0:
                        x_pow.append(x)
                        y_pow.append(y)
                        for p in range(equalize_order - 1):
                            x_pow.append(x_pow[-1]*x)
                            y_pow.append(y_pow[-1]*y)

                    # Set i-th column(s) in the following order: 1, x, y, x^2,
                    # xy, y^2, etc., and the same for j, but with the opposite
                    # sign
                    pofs = 0
                    for o in range(equalize_order + 1):
                        for xp in range(o, -1, -1):
                            yp = o - xp
                            if xp:
                                if yp:
                                    col = x_pow[xp]*y_pow[yp]
                                else:
                                    col = x_pow[xp]
                            else:
                                col = y_pow[yp]
                            if use_sparse:
                                if np.isscalar(col):
                                    col = np.full(npoints, col)
                                a_gen.setdefault(param_offset[i] + pofs, []) \
                                    .append((row, col))
                                a_gen.setdefault(param_offset[j] + pofs, []) \
                                    .append((row, -col))
                            else:
                                a_lsq[row:row+npoints,
                                      param_offset[i]+pofs] = col
                                a_lsq[row:row+npoints,
                                      param_offset[j]+pofs] = -col
                            pofs += 1
                            del col

                    del x_pow, y_pow

                    if equalize_multiplicative:
                        # Apply the already calculated multiplicative model
                        # corrections to overlap data
                        if i:
                            if j:
                                b_lsq[row:row+npoints] = \
                                    d2/transformations[j][0] - \
                                    d1/transformations[i][0]
                            else:
                                b_lsq[row:row+npoints] = \
                                    d2 - d1/transformations[i][0]
                        else:
                            b_lsq[row:row+npoints] = \
                                d2/transformations[j][0] - d1
                    else:
                        b_lsq[row:row+npoints] = d2
                        b_lsq[row:row+npoints] -= d1
                    row += npoints

                if callback is not None:
                    callback(progress + (i + 1)/len(overlaps)*progress_step)
            progress += progress_step

            del overlaps
            gc.collect()

            # Compute the least-squares solution
            if use_sparse:
                # Reconstruct CSC representation
                a_data, a_indices, a_indptr = [], [], [0]
                for j in sorted(a_gen):
                    nonempty_rows = 0
                    for i, d in a_gen[j]:
                        npoints = len(d)
                        a_data.append(d)
                        a_indices.append(np.arange(i, i + npoints))
                        nonempty_rows += npoints
                    a_indptr.append(a_indptr[-1] + nonempty_rows)
                del a_gen
                # noinspection PyTypeChecker
                params = sparse_lstsq(
                    csc_array((np.hstack(a_data), np.hstack(a_indices),
                               np.array(a_indptr)), shape=(m, n)), b_lsq)[0]
                del a_data, a_indices, a_indptr, b_lsq
            else:
                params = lstsq(a_lsq, b_lsq, rcond=None)[0]
                del a_lsq, b_lsq
            gc.collect()
            if equalize_multiplicative:
                for i, ofs in param_offset.items():
                    # Append to multiplicative transformation parameters
                    if i:
                        transformations[i] = np.concatenate(
                            [transformations[i], params[ofs:ofs+npar]])
                    else:
                        transformations[i] = params[ofs:ofs+npar]
            else:
                for i, ofs in param_offset.items():
                    transformations[i] = params[ofs:ofs+npar]

    return transformations


def global_equalize(
        data: Union[np.ndarray, ma.MaskedArray],
        equalize_order: int = 1) -> Union[np.ndarray, ma.MaskedArray]:
    """
    Calculate and subtract the global background equalization transformation

    The background equalization least-squares system is degenerate
    (the coefficient matrix rank is npar (number of parameters per tile) less
    than its number of columns, so in fact we have a family of least-squares
    solutions. This function is called after obtaining the mosaic to choose one
    of these solutions that makes the resulting mosaic background as flat
    as possible by fitting the model to the entire mosaic.

    :param data: mosaic obtained by stacking input tiles
    :param equalize_order: background equalization polynomial order

    :return: flattened mosaic; also `data` is modified in place
    """
    npar = (equalize_order + 1)*(equalize_order + 2)//2
    y, x = np.indices(data.shape)
    x, y, b_lsq = x.ravel(), y.ravel(), data.ravel()
    x_pow, y_pow = [1, x], [1, y]
    for p in range(equalize_order - 1):
        x_pow.append(x_pow[-1]*x)
        y_pow.append(y_pow[-1]*y)
    a_lsq = np.empty((len(x), npar))
    pofs = 0
    for o in range(equalize_order + 1):
        for xp in range(o, -1, -1):
            yp = o - xp
            if xp:
                if yp:
                    col = x_pow[xp]*y_pow[yp]
                else:
                    col = x_pow[xp]
            else:
                col = y_pow[yp]
            a_lsq[:, pofs] = col
            pofs += 1
    params = lstsq(a_lsq, b_lsq, rcond=None)[0]
    pofs = 0
    for o in range(equalize_order + 1):
        for xp in range(o, -1, -1):
            c = params[pofs]
            if c:
                yp = o - xp
                if xp:
                    if yp:
                        col = x_pow[xp]*y_pow[yp]
                    else:
                        col = x_pow[xp]
                else:
                    col = y_pow[yp]
                b_lsq -= c*col
            pofs += 1

    return data
