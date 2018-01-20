"""
Statistics-related functions

:func:`~chauvenet()`: robust outlier rejection using Chauvenet's algorithm.
"""

from __future__ import absolute_import, division, print_function

from numpy import clip, inf, ma, sqrt, zeros
from scipy.special import erf


__all__ = ['chauvenet']


def chauvenet(data, min_vals=10, mean=None, sigma=None, clip_lo=True,
              clip_hi=True):
    """
    Reject outliers using Chauvenet's algorithm

    http://en.wikipedia.org/wiki/Chauvenet%27s_criterion

    :param array_like data: input array or object that can be converted to an
        array, incl. numpy.ma.MaskedArray with some values already rejected;
        for multidimensional data, rejection is done along the 0th axis
    :param int min_vals: minimum number of non-masked values to keep
    :param array_like mean: optional mean value override
    :param array_like sigma: optional standard deviation override
    :param bool clip_lo: reject negative outliers
    :param bool clip_hi: reject positive outliers

    :return: boolean mask array with 1's corresponding to rejected elements;
        same shape as input data
    :rtype: class:`numpy.ndarray`

    >>> import numpy
    >>> data = numpy.zeros([5, 10])
    >>> data[2, 3] = data[4, 5] = 1
    >>> chauvenet(data, min_vals=4).nonzero()
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
