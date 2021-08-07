"""
Statistics-related functions

:func:`~chauvenet()`: robust outlier rejection using Chauvenet's algorithm.
"""

from typing import Optional, Union

from numpy import array, clip, inf, ma, ndarray, sqrt, zeros
from scipy.special import erf


__all__ = ['chauvenet', 'weighted_median']


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
    :param mean: optional mean value override
    :param sigma: optional standard deviation override
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
            m = data.mean(0)
        else:
            m = mean

        if sigma is None:
            s = sqrt(((data - m)**2).sum(0)/(n - 1))
        else:
            s = sigma
        if s.ndim:
            # Keep zero-sigma (= constant across the axis) elements intact
            s[(s <= 0).nonzero()] = inf
        elif s <= 0:
            break

        if clip_lo and clip_hi:
            diff = abs(data - m)
        elif clip_lo:
            # noinspection PyTypeChecker
            diff = clip(m - data, 0, None)
        else:
            # noinspection PyTypeChecker
            diff = clip(data - m, 0, None)
        bad = erf(diff/(s*sqrt(2))) > 1 - 1/(2*n)
        n_bad = bad.sum(0)
        if not n_bad.any() or (n - n_bad < min_vals).all():
            break

        data.mask[bad.nonzero()] = True

    return data.mask


def weighted_median(data: Union[ndarray, list], weights: Union[ndarray, list],
                    period: Optional[float] = None) -> float:
    """
    Calculate weighted median of a 1D data array
    Adapted from: https://gist.github.com/tinybike/d9ff1dad515b66cc0d87

    :param data: input data
    :param weights: input weights, same shape as `data`; not necessarily
        normalized but should be non-negative
    :param period: if specified, handle cyclic quantities (like angles) with
        the given period

    :return: weighted median of data

    >>> weighted_median([1, 2, 3, 4], [4.9, 0.1, 2.5, 2.5])
    2.5
    """
    data, weights = array(data).squeeze(), array(weights).squeeze()
    midpoint = weights.sum()/2

    if period is not None and data.size > 1:
        halfperiod = period/2
        data = data % period
        if (data < halfperiod).any() and (data >= halfperiod).any() and \
                min(data.min(), (period - data).min()) < \
                abs(data - halfperiod).min():
            data[data >= halfperiod] -= period

    if any(weights > midpoint):
        w_median = (data[weights == weights.max()])[0]
    else:
        s_data, s_weights = map(array, zip(*sorted(zip(data, weights))))
        cs_weights = s_weights.cumsum()
        idx = (cs_weights <= midpoint).nonzero()[0][-1]
        if cs_weights[idx] == midpoint:
            w_median = s_data[idx:idx + 2].mean()
        else:
            w_median = s_data[idx + 1]
    return w_median
