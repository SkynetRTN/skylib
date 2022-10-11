"""
Statistics-related functions

:func:`~chauvenet()`: robust outlier rejection using Chauvenet's algorithm.
"""

from typing import Optional, Union

from numpy import (
    array, clip, indices, inf, ma, ndarray, r_, sqrt, vstack, zeros)
from scipy.special import erf


__all__ = ['chauvenet', 'weighted_median', 'weighted_quantile']


def chauvenet(data: ndarray, min_vals: int = 10,
              mean: Optional[Union[ndarray, float, int]] = None,
              sigma: Optional[Union[ndarray, float, int]] = None,
              clip_lo: bool = True, clip_hi: bool = True) -> ndarray:
    """
    Reject outliers using Chauvenet's algorithm

    https://en.wikipedia.org/wiki/Chauvenet%27s_criterion

    :param data: input array or object that can be converted to an array, incl.
        :class:`numpy.ma.MaskedArray` with some values already rejected;
        for multidimensional data, rejection is done along the 0th axis
    :param min_vals: minimum number of non-masked values to keep
    :param mean: optional mean value override; defaults to median of non-masked
        data at each iteration
    :param sigma: optional standard deviation override; defaults to
        the estimate given by the super-simplified version of Robust Chauvenet
        Rejection (Maples et al.)
    :param clip_lo: reject negative outliers
    :param clip_hi: reject positive outliers

    :return: boolean mask array with 1's corresponding to rejected elements;
        same shape as input data

    >>> import numpy
    >>> d = numpy.zeros([5, 10])
    >>> d[2, 3] = d[4, 5] = 1
    >>> chauvenet(d, min_vals=4).nonzero()
    (array([2, 4]), array([3, 5]))
    """
    data = ma.masked_array(data)

    min_vals = max(min_vals, 2)
    n_tot = data.shape[0]
    if n_tot <= min_vals or not clip_lo and not clip_hi:
        # Must keep all values along the given axis, nothing to reject
        return zeros(data.shape, bool)

    if not data.mask.shape:
        data.mask = zeros(data.shape, bool)

    while True:
        n = n_tot - data.mask.sum(0)
        if (n <= min_vals).all():
            break

        if mean is None:
            m = ma.median(data, 0)
        else:
            m = mean

        if clip_lo and clip_hi:
            diff = abs(data - m)
        elif clip_lo:
            # noinspection PyTypeChecker
            diff = clip(m - data, 0, None)
        else:
            # noinspection PyTypeChecker
            diff = clip(data - m, 0, None)

        if sigma is None:
            absdev = vstack([[zeros(diff.shape[1])], diff])
            ndarray.sort(absdev, 0)
            i = (0.683*n).astype(int)
            k = 0.683*(n - 1) % 1
            idx = (i,) + tuple(indices(absdev.shape[1:]))
            idx1 = (i + 1,) + tuple(indices(absdev.shape[1:]))
            s = (absdev[idx] + (absdev[idx1] - absdev[idx])*k)*(1 + 1.7/n)
            if (s <= 0).any():
                # Fall back to normal definition
                s[s <= 0] = sqrt(((data - m)**2).sum(0)/(n - 1))[s <= 0]
        else:
            s = sigma
        if s.ndim:
            # Keep zero-sigma (= constant across the axis) elements intact
            s[(s <= 0).nonzero()] = inf
        elif s <= 0:
            break

        bad = erf(diff/(s*sqrt(2))) > 1 - 0.5/n
        n_bad = bad.sum(0)
        if not n_bad.any() or (n - n_bad < min_vals).all():
            break

        data.mask[bad.nonzero()] = True

    return data.mask


def weighted_quantile(data: Union[ndarray, list],
                      weights: Union[ndarray, list],
                      q: float,
                      period: Optional[float] = None) -> float:
    """
    Calculate weighted quantile of a 1D data array; see
    https://arxiv.org/pdf/1807.05276.pdf, Eqs.17--20

    :param data: input data
    :param weights: input weights, same shape as `data`; not necessarily
        normalized but should be non-negative
    :param q: quantile value, 0 <= `percentile` <= 1
    :param period: if specified, handle cyclic quantities (like angles) with
        the given period

    :return: weighted percentile of data

    >>> weighted_quantile([1, 2, 3, 4], [4.9, 0.1, 2.5, 2.5], 0.683)
    3.049
    """
    data, weights = array(data).squeeze(), array(weights).squeeze()
    sw = q*weights.sum()

    if period is not None and data.size > 1:
        halfperiod = period/2
        data = data % period
        if (data < halfperiod).any() and (data >= halfperiod).any() and \
                min(data.min(), (period - data).min()) < \
                abs(data - halfperiod).min():
            data[data >= halfperiod] -= period

    if any(weights > sw):
        return (data[weights == weights.max()])[0]

    s_data, s_weights = map(array, zip(*sorted(zip(data, weights))))
    cs_weights = s_weights.cumsum()
    sj = r_[0, q*cs_weights + (1 - q)*(r_[0, cs_weights[:-1]])]
    j = (sj >= sw).nonzero()[0][0]
    if j < 2:
        return s_data[0]
    return s_data[j - 2] + (s_data[j - 1] - s_data[j - 2])*(sw - sj[j - 1]) / \
        (sj[j] - sj[j - 1])


def weighted_median(data: Union[ndarray, list], weights: Union[ndarray, list],
                    period: Optional[float] = None) -> float:
    """
    Calculate weighted median of a 1D data array

    :param data: input data
    :param weights: input weights, same shape as `data`; not necessarily
        normalized but should be non-negative
    :param period: if specified, handle cyclic quantities (like angles) with
        the given period

    :return: weighted median of data

    >>> weighted_median([1, 2, 3, 4], [4.9, 0.1, 2.5, 2.5])
    2.0384615384615383
    """
    return weighted_quantile(data, weights, 0.5, period)
