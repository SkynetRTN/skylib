"""
Statistics-related functions

:func:`~chauvenet()`: robust outlier rejection using Chauvenet's algorithm.
"""

import math
from typing import Iterable, Optional, Tuple, Union

import numpy as np
from numba import njit, prange


__all__ = ['chauvenet', 'stddev', 'weighted_median', 'weighted_quantile']


@njit(nogil=True, parallel=True, cache=True)
def stddev(data: np.ndarray, mask: Optional[np.ndarray]):
    """
    Numba implementation of standard deviation of a 1D, 2D, or 3D masked array
    along the 0th axis

    :param data: input array of *residuals* against some mean
    :param mask: optional mask, same shape as `data`

    :return: standard deviation along the 0th axis; values <= 0 are replaced
        with +inf
    """
    if data.ndim == 1:
        sigma: float = 0
        n: int = 0
        for i in range(data.shape[0]):
            if not mask[i]:
                sigma += data[i]**2
                n += 1
        if n > 1:
            sigma /= n - 1
        if sigma > 0:
            sigma = np.sqrt(sigma)
        else:
            sigma = np.inf
    elif data.ndim == 2:
        sigma: np.ndarray = np.zeros(data.shape[1])
        for j in prange(data.shape[1]):
            n = 0
            s = 0
            for i in range(data.shape[0]):
                if not mask[i, j]:
                    s += data[i, j]**2
                    n += 1
            if n > 1:
                s /= n - 1
            if s > 0:
                s = np.sqrt(s)
            else:
                s = np.inf
            sigma[j] = s
    else:
        sigma: np.ndarray = np.zeros(data.shape[1:])
        for j in prange(data.shape[1]):
            for k in range(data.shape[2]):
                n = 0
                s = 0
                for i in range(data.shape[0]):
                    if not mask[i, j, k]:
                        s += data[i, j, k]**2
                        n += 1
                if n > 1:
                    s /= n - 1
                if s > 0:
                    s = np.sqrt(s)
                else:
                    s = np.inf
                sigma[j, k] = s
    return sigma


@njit(nogil=True, cache=True)
def corrfactor(n: int) -> float:
    """
    Empirical RCR correction factor for the given number of points

    :param n: number of points > 1

    :return: correction factor
    """
    if n == 2:
        return 1.76
    if n == 3:
        return 1.59
    if n == 4:
        return 1.53
    if n == 5:
        return 1.31
    return 1 + 2.2212*n**-1.137


# TODO: Enable parallel mode for chauvenet() when Numba 0.57 is out
@njit(nogil=True, cache=True)
def chauvenet(data: np.ndarray, mask: Optional[np.ndarray] = None,
              nu: int = 0, min_vals: int = 10, mean_type: int = 0,
              mean_override: Optional[Union[np.ndarray, float, int]] = None,
              sigma_type: int = 0,
              sigma_override: Optional[Union[np.ndarray, float, int]] = None,
              clip_lo: bool = True, clip_hi: bool = True,
              max_iter: int = 0,
              check_idx: Optional[Union[int, Tuple[int, int],
                                        Tuple[int, int, int]]] = None) \
        -> Tuple[np.ndarray, Union[np.ndarray, float],
                 Union[np.ndarray, float]]:
    """
    Reject outliers using Chauvenet's algorithm or its modification

    https://en.wikipedia.org/wiki/Chauvenet%27s_criterion

    Regular Chauvenet corresponds to `mean_type`=0, `sigma_type`=0 (default).
    `mean_type`=1, `sigma_type`=1 is the simplified version of unweighted
        Robust Chauvenet Rejection (RCR, see Maples et al.).

    :param data: 1D, 2D, or 3D input array; for multidimensional data,
        rejection is done along the 0th axis
    :param mask: optional data mask with already rejected values, same shape
        as `data`; if present, modified in place and returned on output
    :param nu: number of degrees of freedom in Student's distribution; `nu` = 0
        means infinity = Gaussian distribution, nu = 1 => Lorentzian
        distribution; also, `nu` = 2 and 4 are supported; for other values,
        CDF is not analytically invertible
    :param min_vals: minimum number of non-masked values to keep; minimum: 2
    :param mean_type: 0 = mean, 1 = median
    :param mean_override: mean value override, scalar or shape compatible with
        `data`; if present, `mean_type` is ignored
    :param sigma_type: 0 = standard deviation, 1 = absolute deviation at 68-th
        percentile
    :param sigma_override: standard deviation override, scalar or shape
        compatible with `data`; if present, `sigma_type` is ignored
    :param clip_lo: reject negative outliers
    :param clip_hi: reject positive outliers
    :param max_iter: maximum number of rejection iterations; default: no limit
    :param check_idx: for `max_iter` = 0 or > 1, stop looking for more outliers
        as soon as the given data item is identified as an outlier; must be
        scalar for 1D input, 2-tuple for 2D input, and 3-tuple for 3D input

    :return: boolean mask array with 1's corresponding to rejected elements
        (same shape as input data), mu (depending on `mean_type`), and gamma
        (depending on `sigma_type`); mu.shape = gamma.shape = data.shape[1:]

    >>> import numpy
    >>> x = numpy.zeros([5, 10])
    >>> x[2, 3] = x[4, 5] = 1
    >>> chauvenet(x, min_vals=4)[0].nonzero()
    (array([2, 4]), array([3, 5]))
    """
    assert 1 <= data.ndim <= 3
    if mask is None:
        mask = np.zeros(data.shape, np.bool8)

    min_vals = max(min_vals, 2)
    n_tot = data.shape[0]
    n_iter = 0
    while True:
        n = n_tot - mask.sum(0)
        goodmask = ~mask

        if mean_override is None:
            if mean_type == 0:
                if data.ndim == 1:
                    mu = data[goodmask.nonzero()[0]].mean()
                elif data.ndim == 2:
                    mu = np.empty(data.shape[1], float)
                    for i in prange(data.shape[1]):
                        mu[i] = data[goodmask[:, i].nonzero()[0], i].mean()
                else:
                    mu = np.empty(data.shape[1:], float)
                    for i in prange(data.shape[1]):
                        for j in range(data.shape[2]):
                            mu[i, j] = data[goodmask[:, i, j].nonzero()[0],
                                            i, j].mean()
            else:
                if data.ndim == 1:
                    mu = np.median(data[goodmask.nonzero()[0]])
                elif data.ndim == 2:
                    mu = np.empty(data.shape[1], float)
                    for i in prange(data.shape[1]):
                        mu[i] = np.median(data[goodmask[:, i].nonzero()[0], i])
                else:
                    mu = np.empty(data.shape[1:], float)
                    for i in prange(data.shape[1]):
                        for j in range(data.shape[2]):
                            mu[i, j] = np.median(
                                data[goodmask[:, i, j].nonzero()[0], i, j])
        else:
            mu = mean_override

        if clip_lo and clip_hi:
            diff = np.abs(data - mu)
        elif clip_lo:
            # noinspection PyTypeChecker
            diff = np.clip(mu - data, 0, None)
        else:
            # noinspection PyTypeChecker
            diff = np.clip(data - mu, 0, None)

        if sigma_override is None:
            if sigma_type == 0:
                if data.ndim == 1:
                    gamma = stddev(diff, mask)
                elif data.ndim == 2:
                    gamma = np.empty(data.shape[1], float)
                    for i in prange(data.shape[1]):
                        gamma[i] = stddev(diff[:, i], mask[:, i])
                else:
                    gamma = np.empty(data.shape[1:], float)
                    for i in prange(data.shape[1]):
                        for j in range(data.shape[2]):
                            gamma[i, j] = stddev(data[:, i, j], mask[:, i, j])
            else:
                # Quantile for gamma depending on nu
                if nu == 1:
                    q = 0.5
                elif nu == 2:
                    q = 0.577
                elif nu == 4:
                    q = 0.626
                else:
                    q = 0.683

                if data.ndim == 1:
                    good = goodmask.nonzero()[0]
                    if len(good):
                        gamma = np.quantile(diff[good], q)
                    else:
                        gamma = 0
                    if gamma <= 0:
                        gamma = stddev(diff, mask)
                elif data.ndim == 2:
                    gamma = np.empty(data.shape[1], float)
                    for i in prange(data.shape[1]):
                        good = goodmask[:, i].nonzero()[0]
                        if len(good):
                            gamma[i] = np.quantile(diff[good, i], q)
                        else:
                            gamma[i] = 0
                        if gamma[i] <= 0:
                            gamma[i] = stddev(diff[:, i], mask[:, i])
                else:
                    gamma = np.empty(data.shape[1:], float)
                    for i in prange(data.shape[1]):
                        for j in range(data.shape[2]):
                            good = goodmask[:, i, j].nonzero()[0]
                            if len(good):
                                gamma[i, j] = np.quantile(diff[good, i, j], q)
                            else:
                                gamma[i, j] = 0
                            if gamma[i, j] <= 0:
                                gamma[i, j] = stddev(
                                    diff[:, i, j], mask[:, i, j])

                # Apply empirical RCR correction factor
                if data.ndim == 1:
                    gamma *= corrfactor(n)
                elif data.ndim == 2:
                    for i in prange(data.shape[1]):
                        gamma[i] *= corrfactor(n[i])
                else:
                    for i in prange(data.shape[1]):
                        for j in range(data.shape[2]):
                            gamma[i, j] *= corrfactor(n[i, j])
        else:
            gamma = sigma_override

        if max_iter and n_iter >= max_iter or not clip_lo and not clip_hi:
            break
        if data.ndim == 1:
            if n <= min_vals or np.isinf(gamma):
                break
        elif (n <= min_vals).all():
            break

        t = diff/gamma
        if nu == 1:  # Lorentzian
            cdf = 0.5 + np.arctan(t)/np.pi
        elif nu == 2:
            cdf = 0.5 + 0.5/np.sqrt(2)*t/np.sqrt(1 + 0.5*t**2)
        elif nu == 4:
            t **= 2
            d = t/(1 + 0.25*t)
            cdf = 0.5 + 0.375*np.sqrt(d)*(1 - d/12)
        else:  # Gaussian
            t /= np.sqrt(2)
            cdf = np.empty(data.shape, float)
            if data.ndim == 1:
                for i in prange(n_tot):
                    cdf[i] = 0.5*(1 + math.erf(t[i]))
            elif data.ndim == 2:
                for i in prange(n_tot):
                    for j in range(data.shape[1]):
                        cdf[i, j] = 0.5*(1 + math.erf(t[i, j]))
            else:
                for i in prange(n_tot):
                    for j in range(data.shape[1]):
                        for k in range(data.shape[2]):
                            cdf[i, j, k] = 0.5*(1 + math.erf(t[i, j, k]))
        bad = goodmask & (n > min_vals) & (cdf > 1 - 0.25/n)
        n_bad = bad.sum(0)
        if data.ndim == 1 and (not n_bad or n - n_bad < min_vals) or \
                data.ndim > 1 and (not n_bad.any() or
                                   (n - n_bad < min_vals).all()):
            break

        if data.ndim == 1:
            for i in prange(n_tot):
                if bad[i]:
                    mask[i] = True
        elif data.ndim == 2:
            for j in prange(data.shape[1]):
                if n[j] - n_bad[j] >= min_vals:
                    for i in range(n_tot):
                        if bad[i, j]:
                            mask[i, j] = True
        else:
            for j in prange(data.shape[1]):
                for k in range(data.shape[2]):
                    if n[j, k] - n_bad[j, k] >= min_vals:
                        for i in range(n_tot):
                            if bad[i, j, k]:
                                mask[i, j, k] = True

        if check_idx is not None and mask[check_idx]:
            break

        n_iter += 1

    if data.ndim == 1:
        if not np.isfinite(gamma):
            gamma = 0
    else:
        gamma[(~np.isfinite(gamma)).nonzero()] = 0
    return mask, mu, gamma


def weighted_quantile(data: Union[np.ndarray, Iterable],
                      weights: Union[np.ndarray, Iterable],
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
    data, weights = np.array(data).squeeze(), np.array(weights).squeeze()
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

    s_data, s_weights = map(np.array, zip(*sorted(zip(data, weights))))
    cs_weights = s_weights.cumsum()
    sj = np.r_[0, q*cs_weights + (1 - q)*(np.r_[0, cs_weights[:-1]])]
    j = (sj >= sw).nonzero()[0][0]
    if j < 2:
        return s_data[0]
    return s_data[j - 2] + (s_data[j - 1] - s_data[j - 2])*(sw - sj[j - 1]) / \
        (sj[j] - sj[j - 1])


def weighted_median(data: Union[np.ndarray, Iterable],
                    weights: Union[np.ndarray, Iterable],
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
