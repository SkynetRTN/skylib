"""
Geometric pattern matching for alignment without a WCS.

See :func:`~.alignment.apply_transform_stars()`.

The matching algorithm implemented here is based on the substantially modified code from the astroalign package by
Martin Beroiz, with modifications by Prajwel Joseph in the aafitrans package.

The original copyright notice and description follow.

MIT License

Copyright (c) 2016-2019 Martin Beroiz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Modified by Prajwel Joseph
Modified by Vladimir Kouprianov

The following paper should be cited if you use the script in a scientific
publication

Astroalign: A Python module for astronomical image registration.
Beroiz, M., Cabral, J. B., & Sanchez, B.
Astronomy & Computing, Volume 32, July 2020, 100384.

ASTROALIGN aligns stellar images using no WCS information.

It does so by finding similar 3-point asterisms (triangles) in both images and
deducing the affine transformation between them.

General registration routines try to match feature points, using corner
detection routines to make the point correspondence.
These generally fail for stellar astronomical images, since stars have very
little stable structure and so, in general, indistinguishable from each other.

Asterism matching is more robust, and closer to the human way of matching
stellar images.

Astroalign can match images of very different field of view, point-spread
functions, seeing and atmospheric conditions.

A separate copyright is for the RANSAC code:

# Copyright (c) 2004-2007, Andrew D. Straw. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.

#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.

#     * Neither the name of the Andrew D. Straw nor the names of its
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# a PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#
# Modified by Martin Beroiz
# Modified by Prajwel Joseph
# Modified by Vladimir Kouprianov
"""

from itertools import combinations

import numpy as np
from scipy.spatial import KDTree
from numba import njit


__all__ = ['pattern_match']


# Arun and Horn's method.
@njit(nogil=True, cache=True)
def _arun_and_horn(src: np.ndarray, dst: np.ndarray, estimate_scale: bool) -> np.ndarray:
    """Estimate N-D similarity transformation with or without scaling.

    Parameters
    ----------
    src : (M, N) array_like
        Source coordinates.
    dst : (M, N) array_like
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.

    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix.
    """
    num, dim = src.shape

    # Compute mean of src and dst.
    src_mean = np.empty(2, dtype=np.float64)
    src_mean[0] = src[:, 0].mean()
    src_mean[1] = src[:, 1].mean()
    dst_mean = np.empty(2, dtype=np.float64)
    dst_mean[0] = dst[:, 0].mean()
    dst_mean[1] = dst[:, 1].mean()

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    a = dst_demean.T @ src_demean / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.float64)

    t = np.eye(dim + 1, dtype=np.float64)

    u, s, v = np.linalg.svd(a)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(a)
    if rank == 0:
        return np.nan*t
    elif rank == dim - 1:
        if np.linalg.det(u)*np.linalg.det(v) > 0:
            t[:dim, :dim] = u @ v
        else:
            t[:dim, :dim] = u @ np.diag(d) @ v
    else:
        t[:dim, :dim] = u @ np.diag(d) @ v

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0/(src_demean[:, 0].var() + src_demean[:, 1].var())*(s @ d)
        t[:dim, dim] = dst_mean - scale*(t[:dim, :dim].copy() @ src_mean.T.copy())
        t[:dim, :dim] *= scale
    else:
        t[:dim, dim] = dst_mean - (t[:dim, :dim].copy() @ src_mean.T.copy())

    return t


@njit(nogil=True, cache=True)
def _fit_model(source_controlp: np.ndarray, target_controlp: np.ndarray, estimate_scale: bool,
               data: np.ndarray) -> np.ndarray:
    """
    Return the best 2D similarity transform from the points given in data.

    data: N sets of similar corresponding triangles.
        3 indices for a triangle in ref
        and the 3 indices for the corresponding triangle in target;
        arranged in a (N, 3, 2) array.
    """
    d1, d2, d3 = data.shape
    s, d = data.reshape(d1*d2, d3).T
    return _arun_and_horn(source_controlp[s], target_controlp[d], estimate_scale)


@njit(nogil=True, cache=True)
def _apply_mat(coords: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    ndim = matrix.shape[0] - 1
    coords = np.atleast_2d(coords)

    src = np.empty((coords.shape[0], ndim + 1), dtype=np.float64)
    src[:, :ndim] = coords
    src[:, ndim] = 1
    dst = src @ matrix.T

    # below, we will divide by the last dimension of the homogeneous
    # coordinate matrix. In order to avoid division by zero,
    # we replace exact zeros in this column with a very small number.
    dst[dst[:, ndim] == 0, ndim] = np.finfo(np.float64).eps
    # rescale to homogeneous coordinates
    dst[:, :ndim] /= dst[:, ndim : ndim + 1]

    return dst[:, :ndim]


@njit(nogil=True, cache=True)
def _get_error(source_controlp, target_controlp, matrix, data):
    d1, d2, d3 = data.shape
    s, d = data.reshape(d1*d2, d3).T
    resid = (_apply_mat(source_controlp[s], matrix) - target_controlp[d])**2
    error = np.sqrt(resid[:, 0] + resid[:, 1]).reshape(d1, d2)
    for i in range(d1):
        error[i, 0] = error[i].max()
    return error[:, 0]


@njit(nogil=True, cache=True)
def _sidelengths(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> np.ndarray:
    sides = np.empty(3, dtype=np.float64)
    sides[0] = np.linalg.norm(x1 - x2)
    sides[1] = np.linalg.norm(x2 - x3)
    sides[2] = np.linalg.norm(x1 - x3)
    return sides


@njit(nogil=True, cache=True)
def _invariantfeatures(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) \
        -> np.ndarray:
    """Given 3 points x1, x2, x3, return the invariant features for the set."""
    sides = _sidelengths(x1, x2, x3)
    sides.sort()
    features = np.empty(2, dtype=np.float64)
    features[0] = sides[2]/sides[1]
    features[1] = sides[1]/sides[0]
    return features


@njit(nogil=True, cache=True)
def _arrangetriplet(sources: np.ndarray, ind1: int, ind2: int, ind3: int) -> np.ndarray:
    """Order vertex_indices according to length side.

    Order in (a, b, c) form Where:
      a is the vertex defined by L1 & L2
      b is the vertex defined by L2 & L3
      c is the vertex defined by L3 & L1
    and L1 < L2 < L3 are the sides of the triangle
    defined by vertex_indices.
    """
    x1, x2, x3 = sources[ind1], sources[ind2], sources[ind3]

    side_ind = np.array(((ind1, ind2), (ind2, ind3), (ind3, ind1)))
    side_lengths = _sidelengths(x1, x2, x3)
    l1_ind, l2_ind, l3_ind = np.argsort(side_lengths)

    # the most common vertex in the list of vertices for two sides is the
    # point at which they meet.
    triplet = np.empty(3, dtype=np.int64)
    inds = np.empty(4, dtype=np.int64)
    inds[:2] = side_ind[l1_ind]
    inds[2:] = side_ind[l2_ind]
    mn = inds.min()
    triplet[0] = np.bincount(inds - mn).argmax() + mn
    inds[:2] = side_ind[l2_ind]
    inds[2:] = side_ind[l3_ind]
    mn = inds.min()
    triplet[1] = np.bincount(inds - mn).argmax() + mn
    inds[:2] = side_ind[l3_ind]
    inds[2:] = side_ind[l1_ind]
    mn = inds.min()
    triplet[2] = np.bincount(inds - mn).argmax() + mn

    return triplet


def _generate_invariants(sources: np.ndarray, num_nearest_neighbors: int, r_limit: float) -> \
        tuple[np.ndarray, np.ndarray]:
    """Return an array of (unique) invariants derived from the array `sources`.

    Return an array of the indices of `sources` that correspond to each
    invariant, arranged as described in _arrangetriplet.
    """

    inv = []
    triang_vrtx = []
    coordtree = KDTree(sources)
    # The number of nearest neighbors to request (to work with few sources)
    knn = min(len(sources), num_nearest_neighbors)
    for asrc in sources:
        __, indx = coordtree.query(asrc, knn)

        # Generate all possible triangles with the 5 indx provided, and store
        # them with the order (a, b, c) defined in _arrangetriplet
        for cmb in combinations(indx, 3):
            triplet = _arrangetriplet(sources, cmb[0], cmb[1], cmb[2])
            features = _invariantfeatures(*sources[triplet])
            if features.prod() < r_limit:
                triang_vrtx.append(triplet)
                inv.append(features)

    # Remove here all possible duplicate triangles
    inv = np.asarray(inv)
    uniq_ind = np.unique(inv, axis=0, return_index=True)[1]
    inv_uniq = inv[uniq_ind]
    triang_vrtx_uniq = np.array(triang_vrtx)[uniq_ind]

    return inv_uniq, triang_vrtx_uniq


@njit(nogil=True, cache=True)
def _ransac(
        data: np.ndarray, source_controlp: np.ndarray, target_controlp: np.ndarray,
        estimate_scale: bool, thresh: float, min_matches: int, n_samples: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit model parameters to data using the RANSAC algorithm.

    Parameters
    ----------
        data
            (N, 3, 2) array of indices of the potentially matching source and target triangles
        source_controlp
            Array of source control points
        target_controlp
            Array of target control points
        estimate_scale
            When True, use scale-invariant algorithm (similarity transform); otherwise, use Euclidean transform assuming
            same source and target scale
        thresh
            A threshold value to determine when a data point fits a model (in pixels)
        min_matches
            The min number of matches required to assert that a model fits well to data
        n_samples
            The minimum number of data points to fit the model to.
    Returns
    -------
        bestfit: 3x3 matrix of model parameters which best fit the data (or nil if no good model is found)
    """
    good_fit = None
    n_data = data.shape[0]
    all_idxs = np.arange(n_data)

    for iter_i in range(n_data):
        # Partition indices into two random subsets
        maybe_idxs = all_idxs[iter_i : iter_i + n_samples]
        test_idxs = np.concatenate((all_idxs[:iter_i], all_idxs[iter_i + n_samples :]))
        maybeinliers = data[maybe_idxs, :]
        test_points = data[test_idxs, :]
        maybemodel = _fit_model(source_controlp, target_controlp, estimate_scale, maybeinliers)
        test_err = _get_error(source_controlp, target_controlp, maybemodel, test_points)
        # select indices of rows with accepted points
        also_idxs = test_idxs[test_err < thresh]
        alsoinliers = data[also_idxs, :]
        if len(alsoinliers) >= min_matches:
            good_data = np.concatenate((maybeinliers, alsoinliers))
            good_fit = _fit_model(source_controlp, target_controlp, estimate_scale, good_data)
            break

    if good_fit is None:
        raise RuntimeError("List of matching triangles exhausted before an acceptable transformation was found")

    better_fit = good_fit
    prev_good_flags = np.zeros(n_data, dtype=np.bool)
    better_inlier_idxs = np.empty(0, dtype=np.int64)
    for _ in range(100):
        test_err = _get_error(source_controlp, target_controlp, better_fit, data)
        good_flags = test_err < thresh
        if (good_flags == prev_good_flags).all():
            break
        prev_good_flags = good_flags
        better_inlier_idxs = (test_err < thresh).nonzero()[0]
        better_data = data[better_inlier_idxs]
        better_fit = _fit_model(source_controlp, target_controlp, estimate_scale, better_data)

    return better_fit, better_inlier_idxs


def pattern_match(source: np.ndarray, target: np.ndarray, scale_invariant: bool = True, eps: float = 2,
                  r_limit: float = 10, min_matches: int = 5, num_nearest_neighbors: int = 8,
                  kdtree_search_radius: float = 0.02, n_samples: int = 1) -> np.ndarray:
    """
    Find a match between sources from n-element set 1 and sources from m-element set 2; both n and m must be greater
    than 2

    :param source: (n x 2) array of (X, Y) coordinates of sources from set 1
    :param target: (m x 2) array of (X, Y) coordinates of sources from set 2
    :param scale_invariant: enable scale-invariant pairing algorithm
    :param eps: vertex position tolerance (eps > 0)
    :param r_limit: maximum allowed edge ratio (r_limit > 1)
    :param min_matches: minimum number of triangle matches to be found.
    :param num_nearest_neighbors: number of nearest neighbors of a given star (including itself) to construct
        the triangle invariants.
    :param kdtree_search_radius: search radius in invariant feature space
    :param n_samples: minimum number of data points to fit the model to. A value of 1 refers to 1 triangle,
        corresponding to 3 pairs of coordinates.

    :return: n-element integer array of indices of sources in set 2 matching i-th source in set 1; each element is
        either 0 <= j[i] < m or j[i] < 0, the latter indicating that no match could be found for the i-th source
    """
    source_controlp = np.asarray(source)
    target_controlp = np.asarray(target)

    if source.shape[1] != 2 or target.shape[1] != 2:
        raise ValueError("The source and target coordinate lists must have two columns.")

    if len(source_controlp) < 3 or len(target_controlp) < 3:
        # Less than 3 points in the source or target lists means that nothing can be matched
        return np.full(len(source_controlp), -1, dtype=np.int64)

    source_invariants, source_asterisms = _generate_invariants(source_controlp, num_nearest_neighbors, r_limit)
    source_invariant_tree = KDTree(source_invariants)

    target_invariants, target_asterisms = _generate_invariants(target_controlp, num_nearest_neighbors, r_limit)
    target_invariant_tree = KDTree(target_invariants)

    # r = 0.1 is the maximum search distance, 0.1 is an empirical value that
    # returns about the same number of matches than inputs
    # matches_list is a list of lists such that for each element
    # source_invariant_tree.data[i], matches_list[i] is a list of the indices
    # of its neighbors in target_invariant_tree.data
    matches_list = source_invariant_tree.query_ball_tree(target_invariant_tree, r=kdtree_search_radius)

    # matches unravels the previous list of matches into pairs of source and
    # target control point matches.
    # matches is a (N, 3, 2) array. N sets of similar corresponding triangles.
    # 3 indices for a triangle in ref
    # and the 3 indices for the corresponding triangle in target;
    matches = []
    # t1 is an asterism in source, t2 in target
    for t1, t2_list in zip(source_asterisms, matches_list):
        for t2 in target_asterisms[t2_list]:
            matches.append(list(zip(t1, t2)))
    matches = np.array(matches)

    if len(source_controlp) > 3 and len(target_controlp) > 3 or len(matches) > 1:
        _, inlier_ind = _ransac(
            matches,
            source_controlp,
            target_controlp,
            scale_invariant,
            eps,
            min_matches,
            n_samples,
        )
        matches = matches[inlier_ind]

    d1, d2, d3 = matches.shape
    inl_arr = matches.reshape(d1 * d2, d3)
    inl_arr_unique = np.unique(inl_arr, axis=0)
    s, t = inl_arr_unique.T

    match_idx = np.full(len(source_controlp), -1, dtype=np.int64)
    match_idx[s] = t

    return match_idx
