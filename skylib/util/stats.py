"""
Statistics-related functions

:func:`~chauvenet()`: robust outlier rejection using Chauvenet's algorithm.
"""

from typing import Optional, Union

from numpy import (
    arctan, array, clip, inf, isnan, ma, nan, nanquantile, ndarray, ones_like,
    pi, r_, sqrt, zeros)
from scipy.special import erf


__all__ = ['chauvenet', 'weighted_median', 'weighted_quantile']


def chauvenet(data: ndarray, nu: int = 0, min_vals: int = 10,
              mean: Union[str, ndarray, float, int] = 'mean',
              sigma: Union[str, ndarray, float, int] = 'stddev',
              clip_lo: bool = True, clip_hi: bool = True,
              max_iter: int = 0) -> ndarray:
    """
    Reject outliers using Chauvenet's algorithm or its modification

    https://en.wikipedia.org/wiki/Chauvenet%27s_criterion

    Regular Chauvenet corresponds to `mean`="mean", `sigma`="stddev" (default).
    `mean`="median", `sigma`="absdev68" is the simplified version of unweighted
        Robust Chauvenet Rejection (RCR, see Maples et al.).

    :param data: input array or object that can be converted to an array, incl.
        :class:`numpy.ma.MaskedArray` with some values already rejected;
        for multidimensional data, rejection is done along the 0th axis
    :param nu: number of degrees of freedom in Student's distribution; `nu` = 0
        means infinity = Gaussian distribution, nu = 1 => Lorentzian
        distribution; also, `nu` = 2 and 4 are supported; for other values,
        CDF is not analytically invertible
    :param min_vals: minimum number of non-masked values to keep
    :param mean: mean value type ("mean" or "median") or override
    :param sigma: standard deviation type ("stddev" or "absdev68") or override
    :param clip_lo: reject negative outliers
    :param clip_hi: reject positive outliers
    :param max_iter: maximum number of rejection iterations; default: no limit

    :return: boolean mask array with 1's corresponding to rejected elements;
        same shape as input data

    >>> import numpy
    >>> x = numpy.zeros([5, 10])
    >>> x[2, 3] = x[4, 5] = 1
    >>> chauvenet(x, min_vals=4).nonzero()
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

    n_iter = 0
    while not max_iter or n_iter < max_iter:
        n = n_tot - data.mask.sum(0)
        if (n <= min_vals).all():
            break

        if isinstance(mean, str):
            if mean == 'mean':
                m = ma.mean(data, 0)
            else:
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

        if isinstance(sigma, str):
            if sigma == 'stddev':
                # Classic Chauvenet
                gamma = sqrt((diff**2).sum(0)/(n - 1))
            else:
                gamma = nanquantile(diff.filled(nan), 0.683, axis=0)
                if gamma.ndim:
                    gamma[isnan(gamma)] = 0
                elif isnan(gamma):
                    gamma = array(0.0)

                # Apply empirical RCR correction factor
                if gamma.ndim:
                    cf = ones_like(gamma)
                    cf[n == 2] += 0.76
                    cf[n == 3] += 0.59
                    cf[n == 4] += 0.53
                    cf[n == 5] += 0.31
                    cf[n > 5] += (2.2212*n**-1.137)[n > 5]
                elif n == 2:
                    cf = 1.76
                elif n == 3:
                    cf = 1.59
                elif n == 4:
                    cf = 1.53
                elif n == 5:
                    cf = 1.31
                else:
                    cf = 1 + 2.2212*n**-1.137
                gamma *= cf

                if (gamma <= 0).any():
                    # Fall back to normal definition
                    if gamma.ndim:
                        gamma[gamma <= 0] = sqrt((diff**2).sum(0)/(n - 1))[
                            gamma <= 0]
                    else:
                        gamma = sqrt((diff**2).sum()/(n - 1))
        else:
            gamma = sigma
        if gamma.ndim:
            # Keep zero-sigma (= constant across the axis) elements intact
            gamma[(gamma <= 0).nonzero()] = inf
        elif gamma <= 0:
            break

        t = diff/gamma
        if nu == 1:  # Lorentzian
            cdf = 0.5 + arctan(t)/pi
        elif nu == 2:
            cdf = 0.5 + 0.5/sqrt(2)*t/sqrt(1 + 0.5*t**2)
        elif nu == 4:
            d = t**2/(1 + 0.25*t**2)
            cdf = 0.5 + 3/8*sqrt(d)*(1 - 1/12*d)
        else:  # Gaussian
            cdf = 0.5*(1 + erf(t/sqrt(2)))
        bad = (cdf > 1 - 0.25/n) & (n > min_vals)
        n_bad = bad.sum(0)
        if not n_bad.any() or (n - n_bad < min_vals).all():
            break

        data.mask[bad.nonzero()] = True

        n_iter += 1

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
