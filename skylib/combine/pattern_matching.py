"""
Geometric pattern matching for alignment without a WCS.

See :func:`~.alignment.apply_transform_stars()`.

The matching algorithm implemented here is taken from the Apex image analysis package and is based on the classic
triangle pattern match, originally described by Edward J. Groth ("A Pattern-Matching Algorithm for Two-Dimensional
Coordinate Lists" 1986, AJ, 91(5), 1244-1248).

However, the algorithm used here differs in a number of aspects from the generic algorithm by Groth. Instead, it more or
less follows the approach of Valdes et al. (1995) (Valdes, F.G., Campusano, L.E., Velasquez, J.D., and Stetson, P.B.
"FOCAS Automatic Catalog Matching Algorithms" 1995, PASP, 107, 1119-1128). First, triangle similarity is described in
terms of ratios of the longest and the middle edge lengths to the length of the shortest edge, rather than
the longest-to-shortest edge ratio plus cosine of angle between these edges, as in Groth (1986). Thus, a triangle is
described by a single point (u,v) in the 2D triangle space (u = d2/d1, v = d3/d1, where d1 <= d2 <= d3 are edge
lengths). Though formally equivalent, this approach deals with similar quantities (edge ratios), which are directly
comparable. This simplifies the triangle matching criteria and, finally, looks at least more natural.

Second, in our approach, the match criteria (Eqs. (7),(8) in Groth 1986) do not involve separate tolerances for each
triangle; a common tolerance "eps" is used instead. Again, this simplifies the decision process, and has proved to be
sufficient for selecting a correct match in experiments. And, again, here we follow the method of Valdes et al. (1995).

The complete algorithm is as follows:
  1. For the given n-element list of 2D points, a list of all possible triangles is built.
  2. For each triangle, vertices are sorted in such way that the edge connecting vertices 1 and 2 (its length is d1) is
     the shortest, while the edge connecting  vertices 1 and 3 (of length d3) is the longest.
  3. The following quantities are computed: ratios R2 and R3 of edge lengths (R2 = d2/d1, R3 = d3/d1, 1 <= R2 <= R3),
     perimeter P = d1 + d2 + d3 (it retains the original image units of length), and orientation O = +1,-1, which is
     the sign of the dot product (2-1).(3-1).
  4. Triangles are removed, which have either a) the shortest edge d1 smaller than the user-specified rejection factor
     "ksi" (d1 here is dimensionless, divided by the maximum edge length in the list), or b) the longest edge ratio R2
     larger than the user-specified limit "Rlimit". The first criterion may be disabled by setting ksi to 0.
  5. Steps 1-4 are repeated for the second ("reference") set of points.
  6. For each triangle in the first set, a triangle in the second set is found that is the nearest (in the Euclidean
     sense) one in the 2D space of edge ratios (R2,R3), i.e. the one with such (R2',R3') that dR^2 = (R2 - R2')^2
     + (R3 - R3')^2 = min. (One evident disadvantage of this approach is that dR^2 is less sensitive to relative errors
     of R2 than to same errors of R3. Though, this is leveled by inferring an upper limit on R3 (Rlimit), which
     guarantees that R3 is not very much larger than R2.) Match is triggered if the distance dR is less than
     the user-specified match tolerance ("eps").
  7. Exactly as in (Groth 1986), only those matches are left, for which the difference of logarithms of perimeter for
     triangles in a matching pair (which is actually the logarithm of relative scale) does not deviate too much from its
     mean value across all pairs. This step is repeated until no more pairs are removed.
  8. Only pairs with either the same or the opposite orientation, depending on which are in a majority, are left. Steps
     7 and 8 are introduced by Groth (1986) to eliminate false detections.
  9. A matrix of "votes" (in Groth's terminology) is computed: each pair of matching triangles casts a vote for all its
     three vertices. Then, a final match for a pair of points is triggered when the number of votes for this pair is
     greater than the maximum for all pairs, multiplied by the user-specified confidence level (but not less than 2).
     Again, this slightly differs from the original criteria of (Groth 1986).

The algorithm is insensitive to arbitrary relative translation, rescaling, rotation, and flipping (mirroring) of both
sets of points. It is relatively computationally expensive and is recommended for point sets consisting of no more than
several tens of points.

Another version of the same algorithm is not scale-invariant but is more robust in terms of false positives than
the above full version. Thus, it is preferred in situations when the approximate image scale is known.
"""

import numpy as np
from numba import njit, prange


__all__ = ['pattern_match']


# Mapping from edge indices to vertex indices
# edge_to_vertex_mapping = {
#     (0, 1, 2): np.array([0, 1, 2]),
#     (0, 2, 1): np.array([1, 0, 2]),
#     (1, 0, 2): np.array([2, 1, 0]),
#     (1, 2, 0): np.array([1, 2, 0]),
#     (2, 0, 1): np.array([2, 0, 1]),
#     (2, 1, 0): np.array([0, 2, 1]),
# }


@njit(nogil=True, cache=True)
def edges(pos: np.ndarray, ijk: np.ndarray) -> np.ndarray:
    """
    Given a vector of points pos[i] = (Xi,Yi) (i = 0,...,n-1) and a matrix of triangle vertex indices
        ijk[l] = (i[l], j[l], k[l]) (i,j,k = 0,...,n-1, l = 0,...,N-1),
    return a (Nx3x2) array of edge vectors:
        edges[l] = (pj - pi, pk - pj, pk - pi),
    where pi = pos[i[l]], pj = pos[j[l]], pk = pos[k[l]]

    :param pos: vector of 2D points
    :param ijk: vertex indices

    :return: array of edge vectors
    """
    m = ijk.shape[0]
    pi = pos[ijk[:, 0]]
    pj = pos[ijk[:, 1]]
    pk = pos[ijk[:, 2]]
    res = np.empty((m, 3, 2), pos.dtype)
    for i in range(m):
        res[i, 0] = pj[i] - pi[i]
        res[i, 1] = pk[i] - pj[i]
        res[i, 2] = pk[i] - pi[i]
    return res


@njit(nogil=True, cache=True)
def get_tri_indices(n: int) -> np.ndarray:
    """
    Generate a list of triangle vertex indices for the given number of points

    :param n: number of points (n > 0)

    :return: (mx3) matrix of all possible triangles for the given number of points. The first axis enumerates triangles,
        its length is the combinatorial factor m = n(n - 1)(n - 2)/6. The second axis enumerates vertices within
        the corresponding triangle, in ascending order (that is, for a given l, indices[l,0] < indices[l,1] <
        indices[l,2]). E.g. for n = 4, the function returns an integer NumPy array
            [[0 1 2]
             [0 1 3]
             [0 2 3]
             [1 2 3]]
        Thus, for every l, indices[l] = [i,j,k] is the triple of vertex indices for the l-th triangle, while
        I = indices[:,0], J = indices[:,1], and K = indices[:,2] are column-vectors of indices of the 1st, 2nd, and 3rd
        vertex, respectively, for all triangles.

        For n < 3, the function returns an empty (0x0) rank-2 array (matrix).

    Note that the function knows nothing about coordinates of vertices and thus cannot guarantee any particular
    orientation of triangles (clockwise or counter-clockwise).
    """
    if n < 3:
        return np.empty((0, 3), np.int64)
    m = n*(n - 1)*(n - 2)//6
    res = np.empty((m, 3), np.int64)
    l = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                res[l, 0] = i
                res[l, 1] = j
                res[l, 2] = k
                l += 1
    return res


@njit(nogil=True, cache=True)
def tri_list(pos: np.ndarray, ksi: float, r_limit: float, scaled: bool) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For a set of n 2D points, generate all possible triangles formed by the points and return their parameters required
    for subsequent triangle match. According to Groth (1986), the list of triangles is filtered by the two criteria:
        1) the minimum edge should be longer than the user-specified rejection limit ksi;
        2) the maximum-to-minimum edge ratio should be less than r_limit.
    The number of triangles returned well be less or equal to the combinatorial limit N = n(n - 1)(n - 2)/6.

    This function is used for both scale-invariant and fixed-scale versions of the pairing algorithm, depending on
    the value of "scaled". If scale-invariant algorithm is used (scaled = True), then the function returns triangle edge
    ratios, along with the logarithm of perimeter. Otherwise, it returns unscaled edge lengths, in the original pixel
    units.

    :param pos: (nx2) array of points: pos[i] = (Xi,Yi), i = 0,...,n-1
    :param ksi: minimum allowed dimensionless edge length
    :param r_limit: edge ratio limit
    :param scaled: True for scale-invariant algorithm; False for fixed-scale algorithm

    :return:
        If scaled = True:
            A tuple (ijk, logP, R2, R3, orient) of triangle parameters. Each parameter is an N-element array
            (N is the number of triangles):
                - indices (ijk) of the triangle vertices in the input set of points; vertices are ordered in such a way
                  that i-j is the shortest triangle edge, while i-k is the longest one; thus, ijk is a (Nx3) array, and,
                  for l-th triangle, i,j,k = ijk[l]
                - logarithm of perimeter (logP) - a sum of lengths of all three edges, in the original units
                - ratios (R2 and R3) of lengths of the middle (j-k) and the longest (i-k) edges to the shortest edge
                  (i-j)
                - triangle orientation flag (orient), either +1 or -1, which depends on the direction (CW or CCW) from
                  the longest to the shortest edge
        If scaled = False:
            A tuple (ijk, d1, d2, d3, orient), where
                - d1, d2, d3 are edge lengths in the original (pixel) units, d1[i] being the shortest edge and d3[i]
                  the longest one for each triangle
            "ijk" and "orient" are the same as above.
    """
    # Obtain the (Nx3) matrix of vertex indices (i,j,k) for every possible triangle; N = n(n-1)(n-2)/6, where n is
    # the number of input points
    ijk = get_tri_indices(len(pos))
    m = ijk.shape[0]

    # Obtain edge lengths
    d = np.sqrt((edges(pos, ijk)**2).sum(-1))

    # For each triangle, sort edges in ascending order
    order = np.empty(d.shape, np.int64)
    for l in range(m):
        o = order[l] = np.argsort(d[l])
        d[l] = d[l][o]

    # Aliases for d[:,0], d[:,1], d[:,2] will be also useful. Now, for the given l, d1[l] <= d2[l] <= d3[l]
    d1, d2, d3 = np.transpose(d)

    # Compute the maximum edge length within the set of triangles; will be used to convert d1 to dimensionless units
    scale = d3.max()

    # Remove triangles with the minimum edge < elimination threshold ksi, or the ones with the max to min edge ratio too
    # large
    if scaled:
        # Compute edge ratios
        d2 /= d1
        d3 /= d1
        d1 /= scale
        good = (d1 > ksi) & (d3 < r_limit)
        # Compute logarithm of perimeter and rename d to d1 for conformance with scaled=False
        d1 = np.log(d.sum(-1))
    else:
        good = d1 > 0
        for i in range(len(good)):
            if good[i] and (d1[i]/scale < ksi or d3[i]/d1[i] > r_limit):
                good[i] = False
    if not good.any():
        # No valid triangles left after filtering; return empty lists
        return (np.empty((0, 3), np.int64), np.empty((0,), np.float64), np.empty((0,), np.float64),
                np.empty((0,), np.float64), np.empty((0,), np.int64))
    ijk, d1, d2, d3, order = ijk[good], d1[good], d2[good], d3[good], order[good]

    # Reorder vertices according to the previously computed order of edges. After that, i-j will be the shortest edge,
    # while i-k, the longest one.
    edge_to_vertex_mapping = {
        (0, 1, 2): np.array([0, 1, 2]),
        (0, 2, 1): np.array([1, 0, 2]),
        (1, 0, 2): np.array([2, 1, 0]),
        (1, 2, 0): np.array([1, 2, 0]),
        (2, 0, 1): np.array([2, 0, 1]),
        (2, 1, 0): np.array([0, 2, 1]),
    }
    for l in range(len(order)):
        ijk[l] = ijk[l][edge_to_vertex_mapping[(order[l, 0], order[l, 1], order[l, 2])]]

    # Now obtain the (now sorted) edges again to compute the triangle orientation orient[l] = sign(d1[l].d3[l]).
    # I could not go without this step, due to non-trivial sorting of vertices.
    v = edges(pos, ijk)
    orient = np.where((v[:, 0]*v[:, 2]).sum(-1) >= 0, 1, -1)

    # Return the computed triangle parameters: 1) vertex indices, 2) logarithm of perimeter and edge ratios d2, d3, or
    # 2) edge lengths d1,d2,d3, and 3) orientation
    return ijk, d1, d2, d3, orient


@njit(nogil=True, cache=True, parallel=True)
def pattern_match(pos1: np.ndarray, pos2: np.ndarray, scale_invariant: bool = False, eps: float = 0.001,
                  ksi: float = 0.003, r_limit: float = 10, confidence: float = 0.15) -> np.ndarray:
    """
    Find a match between sources from n-element set 1 and sources from m-element set 2; both n and m must be greater
    than 2

    :param pos1: n-element list or (n x 2) array of (X, Y) coordinates of sources from set 1 (usually those from
        the image being aligned)
    :param pos2: m-element list or (m x 2) array of (X, Y) coordinates of sources from set 2 (usually those from
        the reference image)
    :param scale_invariant: enable scale-invariant pairing algorithm
    :param eps: edge ratio tolerance (eps > 0)
    :param ksi: minimum allowed dimensionless edge length (ksi >= 0)
    :param r_limit: maximum allowed edge ratio (r_limit > 1)
    :param confidence: vote confidence level (0 < confidence <= 1)

    :return: n-element integer array of indices of sources in set 2 matching i-th source in set 1; each element is
        either 0 <= j[i] < m or j[i] < 0, the latter indicating that no match could be found for the i-th source
    """
    nn, mm = len(pos1), len(pos2)
    if nn < 3 or mm < 3:
        return np.full(nn, -1)

    # Extract the "X" and "Y" attributes from the input sequences. pos1 refers to the 1st list (= source = plate,
    # of length N); pos2 - to the 2nd list (= reference = catalog, of length M)
    pos1 = np.asarray(pos1, np.float64)
    pos2 = np.asarray(pos2, np.float64)

    if scale_invariant:
        # Obtain the two sets of triangles, with their parameters, for both point lists
        ijk1, log_p1, r1, rr1, orient1 = tri_list(pos1, ksi, r_limit, True)
        ijk2, log_p2, r2, rr2, orient2 = tri_list(pos2, ksi, r_limit, True)

        # For search speedup, sort triangles in the 2nd set by increasing R
        order = np.argsort(rr2)
        ijk2, r2, rr2, log_p2, orient2 = ijk2[order], r2[order], rr2[order], log_p2[order], orient2[order]

        # For each triangle from the 1st set, find the best matching triangle in the 2nd set. "matched" will contain 1's
        # in places where a 1st set triangle has a preliminary 2nd set match; "mapping" will contain indices of the 2nd
        # set triangles for each match.
        n_triangles = len(orient1)
        matched = np.zeros(n_triangles, np.int64)
        mapping = np.zeros(n_triangles, np.int64)
        for l in prange(n_triangles):
            # Find the boundaries of the range [R1 - eps, R1 + eps] in the 2nd list
            left, right = np.searchsorted(rr2, [rr1[l] - eps, rr1[l] + eps])

            # Find match quality for all triangles from the 2nd set within the R range found. In the scale-invariant
            # algorithm, match quality is simply the squared Euclidean distance in the triangle space (u = d2/d1, v = d3/d1)
            # between triangles from the 1st and 2nd sets. In the fixed-scale algorithm, this is the Euclidean distance in
            # the 3D space of edge lengths. Skip if no similar triangles from the 2nd set found.
            qual = (r1[l] - r2[left:right + 1])**2 + (rr1[l] - rr2[left:right + 1])**2
            if not len(qual):
                continue

            # Find the triangle in the second set with the minimum value of match quality
            best_match = np.argmin(qual)

            # If quality estimator for the best matching triangle is below match tolerance (eps), register this in
            # the "matched" array and remember the index of the best matching triangle in the 2nd set
            if qual[best_match] < eps**2:
                matched[l] = 1
                mapping[l] = best_match + left

    else:
        # Duplicating some code to avoid ifs
        ijk1, d1_1, d2_1, d3_1, orient1 = tri_list(pos1, ksi, r_limit, False)
        ijk2, d1_2, d2_2, d3_2, orient2 = tri_list(pos2, ksi, r_limit, False)

        # Convert eps from dimensionless units to pixels, multiplying it by the characteristic edge length
        d2 = np.empty(len(d2_1) + len(d2_2), np.float64)
        d2[:len(d2_1)] = d2_1
        d2[len(d2_1):] = d2_2
        eps *= np.median(d2)

        # For search speedup, sort triangles in the 2nd set by increasing d3
        order = np.argsort(d3_2)
        ijk2, orient2, d1_2, d2_2, d3_2 = ijk2[order], orient2[order], d1_2[order], d2_2[order], d3_2[order]

        n_triangles = len(orient1)
        matched = np.zeros(n_triangles, np.int64)
        mapping = np.zeros(n_triangles, np.int64)
        for l in prange(n_triangles):
            left, right = np.searchsorted(d3_2, [d3_1[l] - eps, d3_1[l] + eps])
            qual = (d1_1[l] - d1_2[left:right + 1])**2 + \
                   (d2_1[l] - d2_2[left:right + 1])**2 + \
                   (d3_1[l] - d3_2[left:right + 1])**2
            if not len(qual):
                continue

            best_match = np.argmin(qual)

            if qual[best_match] < eps**2:
                matched[l] = 1
                mapping[l] = best_match + left

    matched = matched.nonzero()[0]

    # If nothing matched, return immediately
    if not len(matched):
        return np.full(nn, -1, np.int64)

    # From this point, triangles in the 1st set are indexed directly, and only those left that match to some triangle in
    # the 2nd set; "mapping" gives the correspondence between both sets. Also, no need in other parameters related to
    # match quality estimation, except the perimeters (for scale-invariant algorithm) and orientations, which will be
    # used below.
    ijk1, orient1, mapping = ijk1[matched], orient1[matched], mapping[matched]
    if scale_invariant:
        log_p1 = log_p1[matched]

    # Compute the mutual orientation flag: 1 when the triangle in the 1st set and the matched one in the 2nd set are
    # equally oriented (either clockwise or counter-clockwise), 0 otherwise
    same_sense = np.where(orient1*orient2[mapping] > 0, 1, 0)

    # Iteratively eliminate false matches. No need in termination by exceeding the iteration limit, as in (Groth 1986),
    # since iteration always finishes, either when nothing is discarded at the current step, or triangle list is
    # exhausted
    if scale_invariant:
        while len(mapping) > 1:
            # Compute the logarithm of relative scale factor
            log_m = log_p1 - log_p2[mapping]

            # Compute the factor for rejection by logM deviation (Groth 1986)
            n = len(same_sense)
            nplus = same_sense.sum()
            nminus = n - nplus
            mt = abs(nplus - nminus)
            mf = n - mt
            if mf > mt:
                factor = 1
            elif mf < 0.1*mt:
                factor = 3
            else:
                factor = 2

            # Discard false detections; terminate iteration if none found
            good = np.abs(log_m - log_m.mean()) < factor*log_m.std()
            if good.all():
                break
            mapping = mapping[good]
            ijk1, log_p1, same_sense = ijk1[good], log_p1[good], same_sense[good]

    # Discard the possible wrong-sense matches left - only triangles of the same sense (either equally or differently
    # oriented) should remain
    n = len(same_sense)
    nplus = same_sense.sum()
    nminus = n - nplus
    same_sense = (same_sense == int(nplus > nminus)).nonzero()[0]
    mapping, ijk1 = mapping[same_sense], ijk1[same_sense]

    # Generate the (NxM) vote matrix
    votes = np.zeros((nn, mm), np.int64)
    for l in prange(len(mapping)):
        # Now we believe that triangle l from the 1st list is matched to triangle m from the 2nd list. Thus, there is
        # one-to-one correspondence between the same vertices of both triangles, therefore each triangle pair produces
        # a vote for all three vertices
        m = mapping[l]
        for i in range(3):
            votes[ijk1[l, i], ijk2[m, i]] += 1

    # Interpret the vote matrix: for each row, the column with the maximum vote gives the index of a reference point
    # that matches this source point
    match = np.argmax(votes, -1)

    # But discard points that have fewer votes than the given confidence level, or less than 2 votes, as well as
    # ambiguous matches
    mx = max(int(votes.max()*confidence), 2)
    imax = np.argmax(votes, 0)[match]
    for i in prange(nn):
        if votes[i, match[i]] < mx or imax[i] != i:
            match[i] = -1

    return match
