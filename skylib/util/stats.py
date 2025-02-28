"""
Statistics-related functions

:func:`~chauvenet()`: robust outlier rejection using Chauvenet's algorithm.
"""

import math
from typing import Iterable, Optional, Tuple, Union

import numpy as np
from numba import njit, prange


__all__ = [
    'chauvenet', 'weighted_median', 'weighted_quantile',
    'stddev1', 'stddev2', 'stddev3',
    'chauvenet1', 'chauvenet1_parallel', 'chauvenet2', 'chauvenet2_parallel',
    'chauvenet3', 'chauvenet3_parallel',
]


@njit(nogil=True, cache=True)
def stddev1(data: np.ndarray, mask: Optional[np.ndarray]):
    """
    Numba implementation of standard deviation of a 1D masked array

    :param data: input array of *residuals* against some mean
    :param mask: optional mask, same shape as `data`

    :return: standard deviation along the 0th axis; values <= 0 are replaced
        with +inf
    """
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
    return sigma


@njit(nogil=True, parallel=True, cache=True)
def stddev2(data: np.ndarray, mask: Optional[np.ndarray]):
    """
    Numba implementation of standard deviation of a 2D masked array along
    the 0th axis

    :param data: input array of *residuals* against some mean
    :param mask: optional mask, same shape as `data`

    :return: standard deviation along the 0th axis; values <= 0 are replaced
        with +inf
    """
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
    return sigma


@njit(nogil=True, parallel=True, cache=True)
def stddev3(data: np.ndarray, mask: Optional[np.ndarray]):
    """
    Numba implementation of standard deviation of a 3D masked array along
    the 0th axis

    :param data: input array of *residuals* against some mean
    :param mask: optional mask, same shape as `data`

    :return: standard deviation along the 0th axis; values <= 0 are replaced
        with +inf
    """
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
def quantile(data: np.ndarray, q: float) -> float:
    """
    Calculate q-th quantile of 1D data with linear interpolation and correction
    factor as per Maples et al., 2018, ApJS, 238(1), article id.2
    (https://iopscience.iop.org/article/10.3847/1538-4365/aad23d/pdf)

    Does not work for q near 0 and 1!

    :param data: input data
    :param q: quantile (0 to 1)

    :return: q-th quantile of data
    """
    data = data.copy()
    data.sort()
    n = len(data)
    if not n:
        return 0.0
    i = int(np.floor(q*n))
    i_minus = q*(n - 1)
    if n == 2:
        cf = 1.76
    elif n == 3:
        cf = 1.59
    elif n == 4:
        cf = 1.53
    elif n == 5:
        cf = 1.31
    else:
        cf = 1 + 2.2212*n**-1.137
    if i > 0:
        return (data[i-1] + (data[i] - data[i-1]) *
                (i_minus - np.floor(i_minus)))*cf
    return (data[i]*(i_minus - np.floor(i_minus)))*cf


@njit(nogil=True, cache=True)
def ng_cdf(t: np.ndarray, nu: int) -> np.ndarray:
    """
    Return CDF for the given normalized array of residuals for non-Gaussian nu

    :param t: diff/gamma
    :param nu: number of degrees of freedom in Student's distribution;
        supported values: 1 (Lorentzian), 2, 4

    :return: CDF, same shape as input
    """
    if nu == 1:  # Lorentzian
        return 0.5 + np.arctan(t)/np.pi
    if nu == 2:
        return 0.5 + 0.5/np.sqrt(2)*t/np.sqrt(1 + 0.5*t**2)

    t **= 2
    d = t/(1 + 0.25*t)
    return 0.5 + 0.375*np.sqrt(d)*(1 - d/12)


def chauvenet1py(data: np.ndarray, mask: np.ndarray, nu: int, min_vals: int,
                 mean_type: int,
                 mean_override: Optional[Union[np.ndarray, float, int]],
                 sigma_type: int,
                 sigma_override: Optional[Union[np.ndarray, float, int]],
                 clip_lo: bool, clip_hi: bool, max_iter: int,
                 check_idx: Optional[int], q: float) \
        -> Tuple[np.ndarray, float, float]:
    """
    Numba-accelerated implementation of Chauvenet rejection for 1D data

    :param data: 1D input array
    :param mask: data mask with already rejected values, same shape as `data`;
        modified in place and returned on output
    :param nu: number of degrees of freedom in Student's distribution
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
    :param q: quantile for gamma depending on nu

    :return: boolean mask array with 1's corresponding to rejected elements
        (same shape as input data), mu (depending on `mean_type`), and gamma
        (depending on `sigma_type`); mu.shape = gamma.shape = data.shape[1:]
    """
    n_tot = data.shape[0]
    n_iter = 0
    while True:
        # Number of non-rejected values
        n = n_tot
        for i in prange(n_tot):
            if mask[i]:
                n -= 1
        goodmask = ~mask

        if mean_override is None:
            if mean_type == 0:
                if n:
                    mu = data[goodmask].mean()
                else:
                    mu = 0
            else:
                if n:
                    mu = np.median(data[goodmask])
                else:
                    mu = 0
        else:
            mu = mean_override

        dev = data - mu
        if sigma_override is None:
            if sigma_type == 0:
                gamma = stddev1(dev, mask)
            else:
                if goodmask.any():
                    gamma = quantile(np.abs(dev[goodmask]), q)
                else:
                    gamma = 0
                if gamma <= 0:
                    gamma = stddev1(dev, mask)
        else:
            gamma = sigma_override

        if max_iter and n_iter >= max_iter or not clip_lo and not clip_hi or \
                check_idx is not None and mask[check_idx]:
            break
        if n <= min_vals or np.isinf(gamma):
            break

        if clip_lo and clip_hi:
            t = np.abs(dev)
        elif clip_lo:
            # noinspection PyTypeChecker
            t = np.clip(-dev, 0, None)
        else:
            # noinspection PyTypeChecker
            t = np.clip(dev, 0, None)

        t /= gamma
        if nu:
            cdf = ng_cdf(t, nu)
        else:  # Gaussian
            t /= np.sqrt(2)
            cdf = np.empty(data.shape, float)
            for i in prange(n_tot):
                cdf[i] = 0.5*(1 + math.erf(t[i]))
        bad = (goodmask & (n > min_vals) & (cdf > 1 - 0.25/n)).astype(np.int32)
        n_bad = bad.sum(0)
        if not n_bad or n - n_bad < min_vals:
            # Either no more values to reject or fewer values than allowed
            # would remain after rejection
            break

        # Can reject more values
        for i in prange(n_tot):
            if bad[i]:
                mask[i] = True

        n_iter += 1

    if not np.isfinite(gamma):
        gamma = 0

    return mask, mu, gamma


chauvenet1_parallel = njit(nogil=True, parallel=True, cache=True)(chauvenet1py)
chauvenet1 = njit(nogil=True, cache=True)(chauvenet1py)


def chauvenet2py(data: np.ndarray, mask: np.ndarray, nu: int, min_vals: int,
                 mean_type: int,
                 mean_override: Optional[Union[np.ndarray, float, int]],
                 sigma_type: int,
                 sigma_override: Optional[Union[np.ndarray, float, int]],
                 clip_lo: bool, clip_hi: bool, max_iter: int,
                 check_idx: Optional[Tuple[int, int]], q: float) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-accelerated implementation of Chauvenet rejection for 2D data

    :param data: 1D input array; rejection is done along the 0th axis
    :param mask: data mask with already rejected values, same shape as `data`;
        modified in place and returned on output
    :param nu: number of degrees of freedom in Student's distribution
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
        as soon as the given data item is identified as an outlier
    :param q: quantile for gamma depending on nu

    :return: boolean mask array with 1's corresponding to rejected elements
        (same shape as input data), mu (depending on `mean_type`), and gamma
        (depending on `sigma_type`); mu.shape = gamma.shape = data.shape[1:]
    """
    n_tot = data.shape[0]
    n_iter = 0
    while True:
        # Number of non-rejected values
        n = np.full(data.shape[1], n_tot, np.int32)
        for j in prange(data.shape[1]):
            for i in range(n_tot):
                if mask[i, j]:
                    n[j] -= 1
        goodmask = ~mask

        if mean_override is None:
            if mean_type == 0:
                mu = np.empty(data.shape[1], float)
                for i in prange(data.shape[1]):
                    if n[i]:
                        mu[i] = data[goodmask[:, i], i].mean()
                    else:
                        mu[i] = 0
            else:
                mu = np.empty(data.shape[1], float)
                for i in prange(data.shape[1]):
                    if n[i]:
                        mu[i] = np.median(data[goodmask[:, i], i])
                    else:
                        mu[i] = 0
        else:
            mu = mean_override

        dev = data.copy()
        for j in prange(data.shape[1]):
            muj = mu[j]
            for i in range(n_tot):
                dev[i, j] -= muj
        if sigma_override is None:
            if sigma_type == 0:
                gamma = np.empty(data.shape[1], float)
                for i in prange(data.shape[1]):
                    gamma[i] = stddev1(dev[:, i], mask[:, i])
            else:
                gamma = np.empty(data.shape[1], float)
                for i in prange(data.shape[1]):
                    if goodmask[:, i].any():
                        gamma[i] = quantile(
                            np.abs(dev[goodmask[:, i], i]), q)
                    else:
                        gamma[i] = 0
                    if gamma[i] <= 0:
                        gamma[i] = stddev1(dev[:, i], mask[:, i])
        else:
            gamma = sigma_override

        if max_iter and n_iter >= max_iter or not clip_lo and not clip_hi or \
                check_idx is not None and mask[check_idx]:
            break
        if (n <= min_vals).all():
            break

        if clip_lo and clip_hi:
            t = np.abs(dev)
        elif clip_lo:
            # noinspection PyTypeChecker
            t = np.clip(-dev, 0, None)
        else:
            # noinspection PyTypeChecker
            t = np.clip(dev, 0, None)

        for j in prange(data.shape[1]):
            gammaj = gamma[j]
            for i in range(n_tot):
                t[i, j] /= gammaj
        if nu:
            cdf = ng_cdf(t, nu)
        else:  # Gaussian
            t /= np.sqrt(2)
            cdf = np.empty(data.shape, float)
            for i in prange(n_tot):
                for j in range(data.shape[1]):
                    cdf[i, j] = 0.5*(1 + math.erf(t[i, j]))
        bad = (goodmask & (n > min_vals) & (cdf > 1 - 0.25/n)).astype(np.int32)
        for j in prange(data.shape[1]):
            nj = n[j]
            if nj <= min_vals:
                for i in range(n_tot):
                    bad[i, j] = 0
                continue
            for i in range(n_tot):
                if cdf[i, j] <= 1 - 0.25/nj:
                    bad[i, j] = 0
        n_bad = bad.sum(0)
        if ((n_bad == 0) | (n - n_bad < min_vals)).all():
            # Either no more values to reject or fewer values than allowed
            # would remain after rejection
            break

        # Can reject more values
        for j in prange(data.shape[1]):
            if n[j] - n_bad[j] >= min_vals:
                for i in range(n_tot):
                    if bad[i, j]:
                        mask[i, j] = True

        n_iter += 1

    for j in prange(data.shape[1]):
        if not np.isfinite(gamma[j]):
            gamma[j] = 0

    return mask, mu, gamma


chauvenet2_parallel = njit(nogil=True, parallel=True, cache=True)(chauvenet2py)
chauvenet2 = njit(nogil=True, cache=True)(chauvenet2py)


def chauvenet3py(data: np.ndarray, mask: np.ndarray, nu: int, min_vals: int,
                 mean_type: int,
                 mean_override: Optional[Union[np.ndarray, float, int]],
                 sigma_type: int,
                 sigma_override: Optional[Union[np.ndarray, float, int]],
                 clip_lo: bool, clip_hi: bool, max_iter: int,
                 check_idx: Optional[Tuple[int, int, int]], q: float) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-accelerated implementation of Chauvenet rejection for 3D data

    :param data: 1D input array; rejection is done along the 0th axis
    :param mask: data mask with already rejected values, same shape as `data`;
        modified in place and returned on output
    :param nu: number of degrees of freedom in Student's distribution
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
        as soon as the given data item is identified as an outlier
    :param q: quantile for gamma depending on nu

    :return: boolean mask array with 1's corresponding to rejected elements
        (same shape as input data), mu (depending on `mean_type`), and gamma
        (depending on `sigma_type`); mu.shape = gamma.shape = data.shape[1:]
    """
    n_tot = data.shape[0]
    n_iter = 0
    while True:
        # Number of non-rejected values
        n = np.full(data.shape[1:], n_tot, np.int32)
        for j in prange(data.shape[1]):
            for k in range(data.shape[2]):
                for i in range(n_tot):
                    if mask[i, j, k]:
                        n[j, k] -= 1
        goodmask = ~mask

        if mean_override is None:
            if mean_type == 0:
                mu = np.empty(data.shape[1:], float)
                for i in prange(data.shape[1]):
                    for j in range(data.shape[2]):
                        if n[i, j]:
                            mu[i, j] = data[goodmask[:, i, j], i, j].mean()
                        else:
                            mu[i, j] = 0
            else:
                mu = np.empty(data.shape[1:], float)
                for i in prange(data.shape[1]):
                    for j in range(data.shape[2]):
                        if n[i, j]:
                            mu[i, j] = np.median(
                                data[goodmask[:, i, j], i, j])
                        else:
                            mu[i, j] = 0
        else:
            mu = mean_override

        dev = data.copy()
        for j in prange(data.shape[1]):
            for k in range(data.shape[2]):
                mujk = mu[j, k]
                for i in range(n_tot):
                    dev[i, j, k] -= mujk
        if sigma_override is None:
            if sigma_type == 0:
                gamma = np.empty(data.shape[1:], float)
                for i in prange(data.shape[1]):
                    for j in range(data.shape[2]):
                        gamma[i, j] = stddev1(dev[:, i, j], mask[:, i, j])
            else:
                gamma = np.empty(data.shape[1:], float)
                for i in prange(data.shape[1]):
                    for j in range(data.shape[2]):
                        if goodmask[:, i, j].any():
                            gamma[i, j] = quantile(
                                np.abs(dev[goodmask[:, i, j], i, j]), q)
                        else:
                            gamma[i, j] = 0
                        if gamma[i, j] <= 0:
                            gamma[i, j] = stddev1(
                                dev[:, i, j], mask[:, i, j])
        else:
            gamma = sigma_override

        if max_iter and n_iter >= max_iter or not clip_lo and not clip_hi or \
                check_idx is not None and mask[check_idx]:
            break
        if (n <= min_vals).all():
            break

        if clip_lo and clip_hi:
            t = np.abs(dev)
        elif clip_lo:
            # noinspection PyTypeChecker
            t = np.clip(-dev, 0, None)
        else:
            # noinspection PyTypeChecker
            t = np.clip(dev, 0, None)

        for j in prange(data.shape[1]):
            for k in range(data.shape[2]):
                gammajk = gamma[j, k]
                for i in range(n_tot):
                    t[i, j, k] /= gammajk
        if nu:
            cdf = ng_cdf(t, nu)
        else:  # Gaussian
            t /= np.sqrt(2)
            cdf = np.empty(data.shape, float)
            for i in prange(n_tot):
                for j in range(data.shape[1]):
                    for k in range(data.shape[2]):
                        cdf[i, j, k] = 0.5*(1 + math.erf(t[i, j, k]))
        bad = goodmask.astype(np.int32)
        for j in prange(data.shape[1]):
            for k in range(data.shape[2]):
                njk = n[j, k]
                if njk <= min_vals:
                    for i in range(n_tot):
                        bad[i, j, k] = 0
                    continue
                for i in range(n_tot):
                    if cdf[i, j, k] <= 1 - 0.25/njk:
                        bad[i, j, k] = 0

        n_bad = bad.sum(0)
        if ((n_bad == 0) | (n - n_bad < min_vals)).all():
            # Either no more values to reject or fewer values than allowed
            # would remain after rejection
            break

        # Can reject more values
        for j in prange(data.shape[1]):
            for k in range(data.shape[2]):
                if n[j, k] - n_bad[j, k] >= min_vals:
                    for i in range(n_tot):
                        if bad[i, j, k]:
                            mask[i, j, k] = True

        n_iter += 1

    for j in prange(data.shape[1]):
        for k in range(data.shape[2]):
            if not np.isfinite(gamma[j, k]):
                gamma[j, k] = 0

    return mask, mu, gamma


chauvenet3_parallel = njit(nogil=True, parallel=True, cache=True)(chauvenet3py)
chauvenet3 = njit(nogil=True, cache=True)(chauvenet3py)


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
    ndim = data.ndim
    assert 1 <= ndim <= 3
    if mask is None:
        mask = np.zeros(data.shape, np.bool)

    min_vals = max(min_vals, 2)

    # Quantile for gamma depending on nu
    if nu == 1:
        q = 0.5
    elif nu == 2:
        q = 0.577
    elif nu == 4:
        q = 0.626
    else:
        q = 0.683
    # Decrease the probability that Nq is a whole number
    q += 1e-7

    return (chauvenet1_parallel, chauvenet2_parallel,
            chauvenet3_parallel)[ndim - 1](
        np.ascontiguousarray(data).astype(np.float64), mask, nu, min_vals,
        mean_type, mean_override, sigma_type, sigma_override, clip_lo, clip_hi,
        max_iter, check_idx, q)


def weighted_quantile(data: Union[np.ndarray, Iterable],
                      weights: Union[np.ndarray, Iterable],
                      q: float,
                      period: Optional[float] = None) -> float:
    """
    Calculate weighted quantile of a 1D data array; see
    https://iopscience.iop.org/article/10.3847/1538-4365/aad23d/pdf, Eqs.17--20

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
