"""
Image alignment.

:func:`~apply_transform_wcs()`: align an image based on WCS.
:func:`~apply_transform_stars()`: align an image based on pixel coordinates of
1, 2, or more stars.
"""

from typing import List as TList, Optional, Tuple, Union

import numpy as np
import scipy.ndimage as nd
from astropy.wcs import WCS
import cv2 as cv


__all__ = [
    'get_transform_stars', 'get_transform_wcs', 'get_transform_features',
    'get_transform_pixel', 'apply_transform',
]


def match_ref_shape(img: Union[np.ndarray, np.ma.MaskedArray],
                    ref_width: int, ref_height: int) \
        -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Pad the image to match the reference image shape

    :param img: input image
    :param ref_width: reference image width
    :param ref_height: reference image height

    :return: padded image, its mask as float32, and the original image average
    """
    h, w = img.shape
    avg = float(img.mean())

    if w < ref_width or h < ref_height:
        new_img = np.full(
            [max(h, ref_height), max(w, ref_width)], avg, img.dtype)
        mask = np.ones(new_img.shape, bool)
        if isinstance(img, np.ma.MaskedArray) and img.mask.any():
            new_img[:h, :w] = img.data
            mask[:h, :w] = img.mask
        else:
            new_img[:h, :w] = img
            mask[:h, :w] = False
        img = np.ma.MaskedArray(new_img, mask)

    if isinstance(img, np.ma.MaskedArray) and img.mask.any():
        # scipy.ndimage does not handle masked arrays; fill masked values with
        # global mean and mask them after transformation
        if img.mask.shape:
            mask = img.mask.astype(np.float32)
        else:
            mask = np.full(img.shape, img.mask, np.float32)
        img = img.filled(avg)
    else:
        mask = np.zeros(img.shape, np.float32)

    return img, mask, avg


def get_transform_stars(src_stars: Union[TList[Tuple[float, float]],
                                         np.ndarray],
                        dst_stars: Union[TList[Tuple[float, float]],
                                         np.ndarray],
                        enable_rot: bool = True,
                        enable_scale: bool = True,
                        enable_skew: bool = True) \
        -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Calculate the alignment transformation based on pixel coordinates of one
    or more stars

    :param src_stars: list of (X, Y) coordinates of one or more alignment stars
        in the image being aligned, 1-based
    :param dst_stars: list of (X, Y) coordinates of the same stars as in
        `src_stars` in the reference image, 1-based
    :param enable_rot: allow rotation transformation for >= 2 points
    :param enable_scale: allow scaling transformation for >= 2 points
    :param enable_skew: allow skew transformation for >= 2 points; ignored
        and set to False if `enable_rot`=False or `enable_scale`=False

    :return: 2x2 linear transformation matrix and offset vector [dy, dx]
    """
    nref = min(len(src_stars), len(dst_stars))

    # Convert to 0-based, as needed by SciPy
    src_x, src_y = np.transpose(src_stars[:nref]) - 1
    dst_x, dst_y = np.transpose(dst_stars[:nref]) - 1

    if not enable_rot or not enable_scale:
        enable_skew = False

    if nref == 1 or not enable_rot and not enable_scale:
        # Pure shift
        return None, np.array([src_y[0] - dst_y[0], src_x[0] - dst_x[0]])

    for i in range(nref - 1):
        x0, y0 = dst_x[i], dst_y[i]
        for j in range(i + 1, nref):
            if (dst_x[j] - x0)**2 + (dst_y[j] - y0)**2 <= 0:
                raise ValueError(
                    'Two or more reference stars have equal coordinates')

    if enable_rot and not enable_scale:
        # 2+ stars, shift + rotation, overdetermined solution:
        #   [ src_y ]   [  A b ] [ dst_y ]   [ dy ]
        #   [ src_x ] = [ -b A ] [ dst_x ] + [ dx ],
        #   A = cos(phi), b = sin(phi)
        src_dy, src_dx = src_y[:-1] - src_y[1:], src_x[:-1] - src_x[1:]
        dst_dy, dst_dx = dst_y[:-1] - dst_y[1:], dst_x[:-1] - dst_x[1:]
        phi = np.arctan2(
            src_dy*dst_dx - src_dx*dst_dy,
            src_dy*dst_dy + src_dx*dst_dx).mean()
        a, b = np.cos(phi), np.sin(phi)
        return np.array([[a, b], [-b, a]]), np.array(
            [(src_y - a*dst_y - b*dst_x).mean(),
             (src_x - a*dst_x + b*dst_y).mean()])

    if nref == 2:
        # Can do only partial affine transform, always exact solution
        if not enable_rot:
            # 2 stars, shift + scale:
            #   [ src_y ]   [ A 0 ] [ dst_y ]   [ dy ]
            #   [ src_x ] = [ 0 B ] [ dst_x ] + [ dx ]
            dst_dy, dst_dx = dst_y[0] - dst_y[1], dst_x[0] - dst_x[1]
            if dst_dy:
                a = (src_y[0] - src_y[1])/dst_dy
            else:
                a = 1
            if dst_dx:
                b = (src_x[0] - src_x[1])/dst_dx
            else:
                b = 1
            return np.array([[a, 0], [0, b]]), np.array(
                [src_y[0] - a*dst_y[0], src_x[0] - b*dst_x[0]])

        # 2 stars, shift + rotation + uniform scale:
        #   [ src_y ]   [  A B ] [ dst_y ]   [ dy ]
        #   [ src_x ] = [ -B A ] [ dst_x ] + [ dx ]
        src_dy, src_dx = src_y[0] - src_y[1], src_x[0] - src_x[1]
        dst_dy, dst_dx = dst_y[0] - dst_y[1], dst_x[0] - dst_x[1]
        d = dst_dx**2 + dst_dy**2
        a = (src_dy*dst_dy + src_dx*dst_dx)/d
        b = (src_dy*dst_dx - src_dx*dst_dy)/d
        return np.array([[a, b], [-b, a]]), np.array(
            [src_y[0] - dst_y[0]*a - dst_x[0]*b,
             src_x[0] - dst_x[0]*a + dst_y[0]*b])

    if not enable_rot:
        # 3+ stars, shift + scale:
        #   [ src_y ]   [ A 0 ] [ dst_y ]   [ dy ]
        #   [ src_x ] = [ 0 B ] [ dst_x ] + [ dx ]
        py = np.linalg.lstsq(
            np.transpose([dst_y, np.ones(nref)]), src_y, rcond=None)[0]
        px = np.linalg.lstsq(
            np.transpose([dst_x, np.ones(nref)]), src_x, rcond=None)[0]
        return np.array([[py[0], 0], [0, px[0]]]), np.array([py[1], px[1]])

    if not enable_skew:
        # 3+ stars, shift + rotation + scale:
        #   [ src_y ]   [ A B ] [ dst_y ]   [ dy ]
        #   [ src_x ] = [ c D ] [ dst_x ] + [ dx ],
        #   c = -B*D/A if A != 0 or
        #   [ src_y ]   [ 0 B ] [ dst_y ]   [ dy ]
        #   [ src_x ] = [ C 0 ] [ dst_x ] + [ dx ]  otherwise
        a, b, dy = np.linalg.lstsq(
            np.transpose([dst_y, dst_x, np.ones(nref)]), src_y, rcond=None)[0]
        if a:
            d, dx = np.linalg.lstsq(
                np.transpose([-b/a*dst_y + dst_x, np.ones(nref)]), src_x,
                rcond=None)[0]
            c = -b/a*d
        else:
            c, dx = np.linalg.lstsq(
                np.transpose([dst_y, np.ones(nref)]), src_x, rcond=None)[0]
            d = 0
        return np.array([[a, b], [c, d]]), np.array([dy, dx])

    # 3+ stars, full affine transform:
    #   [ src_y ]   [ A B ] [ dst_y ]   [ dy ]
    #   [ src_x ] = [ C D ] [ dst_x ] + [ dx ]
    a = np.transpose([dst_y, dst_x, np.ones(nref)])
    py = np.linalg.lstsq(a, src_y, rcond=None)[0]
    px = np.linalg.lstsq(a, src_x, rcond=None)[0]
    return np.array([py[:2], px[:2]]), np.array([py[2], px[2]])


wcs_grid = {
    1: (np.array([1/2]),
        np.array([1/2])),
    2: (np.array([1/3, 2/3]),
        np.array([1/2, 1/2])),
    3: (np.array([1/4, 1/2, 3/4]),
        np.array([1/3, 2/3, 1/3])),
    4: (np.array([1/3, 2/3, 1/3, 2/3]),
        np.array([1/3, 1/3, 2/3, 2/3])),
    5: (np.array([1/3, 2/3, 1/3, 2/3, 1/2]),
        np.array([1/3, 1/3, 2/3, 2/3, 1/2])),
    6: (np.array([1/4, 1/2, 3/4, 1/4, 1/2, 3/4]),
        np.array([1/3, 1/3, 1/3, 2/3, 2/3, 2/3])),
    7: (np.array([1/4, 1/2, 3/4, 1/4, 1/2, 3/4, 1/2]),
        np.array([1/3, 1/3, 1/3, 2/3, 2/3, 2/3, 1/2])),
    8: (np.array([1/4, 1/2, 3/4, 1/4, 1/2, 3/4, 1/3, 2/3]),
        np.array([1/3, 1/3, 1/3, 2/3, 2/3, 2/3, 1/2, 1/2])),
    9: (np.array([1/4, 1/2, 3/4, 1/4, 1/2, 3/4, 1/4, 1/2, 3/4]),
        np.array([1/4, 1/4, 1/4, 1/2, 1/2, 1/2, 3/4, 3/4, 3/4])),
}


def get_transform_wcs(src_wcs: WCS, dst_wcs: WCS,
                      grid_points: int = 0,
                      enable_rot: bool = True,
                      enable_scale: bool = True,
                      enable_skew: bool = True) \
        -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Calculate the alignment transformation based on WCS

    :param src_wcs: WCS of image being aligned
    :param dst_wcs: reference image WCS
    :param grid_points: number of grid points for WCS interpolation::
        0: full affine transform using WCS calculated for each pixel (slow)
        1: offset-only alignment using central pixel
        2: shift + rotation + uniform scale (2-star) alignment using two points
        >= 3: full affine transform using the given number of fake "alignment
            stars" generated from the WCS
    :param enable_rot: allow rotation transformation for `grid_points` >= 2
    :param enable_scale: allow scaling transformation for `grid_points` >= 2
    :param enable_skew: allow skew transformation for `grid_points` >= 2;
        ignored and set to False if `enable_rot`=False or `enable_scale`=False

    :return: 2x2 linear transformation matrix and offset vector [dy, dx]
    """
    ref_width, ref_height = dst_wcs.array_shape
    if grid_points <= 0 or grid_points >= ref_width*ref_height:
        grid_points = ref_width*ref_height

    # Calculate fake alignment stars by sampling WCS on a grid
    try:
        # Special grid for small number of points
        if ref_width >= ref_height:
            dst_x, dst_y = wcs_grid[grid_points]
        else:
            dst_y, dst_x = wcs_grid[grid_points]
        # Cannot multiply in place to avoid damaging wcs_grid
        dst_x = dst_x*ref_width
        dst_y = dst_y*ref_height
    except KeyError:
        # Generate uniform grid; not necessarily the requested number of points
        nx, ny = int(np.sqrt(grid_points*ref_width/ref_height) + 0.5), \
                 int(np.sqrt(grid_points*ref_height/ref_width) + 0.5)
        dst_x, dst_y = np.mgrid[
            :ref_width:(nx + 2)*1j, :ref_height:(ny + 2)*1j]
        dst_x, dst_y = dst_x[1:-1].ravel(), dst_y[1:-1].ravel()

    # Convert to 1-based (required by apply_transform_stars()
    dst_x += 1
    dst_y += 1

    a, d = dst_wcs.all_pix2world(dst_x, dst_y, 1)
    src_x, src_y = src_wcs.all_world2pix(a, d, 1, quiet=True)

    return get_transform_stars(
        np.transpose([src_x, src_y]), np.transpose([dst_x, dst_y]),
        enable_rot=enable_rot, enable_scale=enable_scale,
        enable_skew=enable_skew)


def get_transform_features(img: Union[np.ndarray, np.ma.MaskedArray],
                           ref_img: Union[np.ndarray, np.ma.MaskedArray],
                           enable_rot: bool = True,
                           enable_scale: bool = True,
                           enable_skew: bool = True,
                           algorithm: str = 'AKAZE',
                           ratio_threshold: float = 0.7,
                           detect_edges: bool = False,
                           percentile_min: float = 10,
                           percentile_max: float = 99,
                           **kwargs) \
        -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Calculate the alignment transformation based on feature similarity

    :param img: input image as 2D NumPy array
    :param ref_img: reference image
    :param enable_rot: allow rotation transformation for >= 2 points
    :param enable_scale: allow scaling transformation for >= 2 points
    :param enable_skew: allow skew transformation for >= 2 points; ignored
        and set to False if `enable_rot`=False or `enable_scale`=False
    :param algorithm: feature detection algorithm: "AKAZE" (default), "BRISK",
        "KAZE", "ORB", "SIFT"; more are available if OpenCV contribution
        modules are installed: "SURF" (patented, not always available);
        for more info, see OpenCV docs
    :param ratio_threshold: Lowe's feature match test factor
    :param detect_edges: apply edge detection before feature extraction
    :param percentile_min: lower percentile for conversion to 8 bit
    :param percentile_max: upper percentile for conversion to 8 bit
    :param kwargs: extra feature detector-specific keyword arguments

    :return: 2x2 linear transformation matrix and offset vector [dy, dx]
    """
    if detect_edges:
        img = np.hypot(
            nd.sobel(img, 0, mode='nearest'),
            nd.sobel(img, 1, mode='nearest')
        )
        ref_img = np.hypot(
            nd.sobel(ref_img, 0, mode='nearest'),
            nd.sobel(ref_img, 1, mode='nearest')
        )

    # Convert both images to [0,255) grayscale
    src_img = img
    if percentile_min <= 0 and percentile_max >= 100:
        mn, mx = src_img.min(), src_img.max()
    elif percentile_min <= 0:
        mn = src_img.min()
        if isinstance(src_img, np.ma.MaskedArray):
            mx = np.nanpercentile(src_img.filled(np.nan), percentile_max)
        else:
            mx = np.nanpercentile(src_img, percentile_max)
    elif percentile_max >= 100:
        if isinstance(src_img, np.ma.MaskedArray):
            mn = np.nanpercentile(src_img.filled(np.nan), percentile_min)
        else:
            mn = np.nanpercentile(src_img, percentile_min)
        mx = src_img.max()
    else:
        if isinstance(src_img, np.ma.MaskedArray):
            mn, mx = np.nanpercentile(
                src_img.filled(np.nan), [percentile_min, percentile_max])
        else:
            mn, mx = np.nanpercentile(
                src_img, [percentile_min, percentile_max])
    if mn >= mx:
        raise ValueError('Empty image')
    if not isinstance(src_img, np.ma.MaskedArray):
        src_img = np.ma.masked_invalid(src_img)
    src_img = (np.clip((src_img.filled(mn) - mn)/(mx - mn), 0, 1)*255 + 0.5) \
        .astype(np.uint8)

    dst_img = ref_img
    if percentile_min <= 0 and percentile_max >= 100:
        mn, mx = dst_img.min(), dst_img.max()
    elif percentile_min <= 0:
        mn = dst_img.min()
        if isinstance(dst_img, np.ma.MaskedArray):
            mx = np.nanpercentile(dst_img.filled(np.nan), percentile_max)
        else:
            mx = np.nanpercentile(dst_img, percentile_max)
    elif percentile_max >= 100:
        if isinstance(dst_img, np.ma.MaskedArray):
            mn = np.nanpercentile(dst_img.filled(np.nan), percentile_min)
        else:
            mn = np.nanpercentile(dst_img, percentile_min)
        mx = dst_img.max()
    else:
        if isinstance(dst_img, np.ma.MaskedArray):
            mn, mx = np.nanpercentile(
                dst_img.filled(np.nan), [percentile_min, percentile_max])
        else:
            mn, mx = np.nanpercentile(
                dst_img, [percentile_min, percentile_max])
    if mn >= mx:
        raise ValueError('Empty reference image')
    if not isinstance(dst_img, np.ma.MaskedArray):
        dst_img = np.ma.masked_invalid(dst_img)
    dst_img = (np.clip((dst_img.filled(mn) - mn)/(mx - mn), 0, 1)*255 + 0.5) \
        .astype(np.uint8)

    # Extract features
    if algorithm == 'AKAZE':
        fe = cv.AKAZE_create(**kwargs)
    elif algorithm == 'BRISK':
        fe = cv.BRISK_create(**kwargs)
    elif algorithm == 'KAZE':
        fe = cv.KAZE_create(**kwargs)
    elif algorithm == 'ORB':
        fe = cv.ORB_create(**kwargs)
    elif algorithm == 'SIFT':
        fe = cv.SIFT_create(**kwargs)
    elif algorithm == 'SURF':
        fe = cv.xfeatures2d.SURF_create(**kwargs)
    else:
        raise ValueError(
            'Unknown feature detection algorithm "{}"'.format(algorithm))
    kp1, des1 = fe.detectAndCompute(src_img, None)
    kp2, des2 = fe.detectAndCompute(dst_img, None)

    # Cross-match features
    matcher = cv.BFMatcher(
        cv.NORM_L2 if algorithm in ('KAZE', 'SIFT', 'SURF')
        else cv.NORM_HAMMING2
        if algorithm == 'ORB' and kwargs.get('WTA_K', 2) in (3, 4)
        else cv.NORM_HAMMING)
    matches = matcher.knnMatch(des1, des2, k=2)

    # Filter matches using Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold*n.distance:
            good_matches.append(m)
    if not good_matches:
        raise ValueError('No matching features found')

    # Keep only one unique match for each reference feature
    ref_idx = np.array([m.trainIdx for m in good_matches])
    distances = np.array([m.distance for m in good_matches])
    min_idx = ref_idx.min(initial=0)
    num_matches = np.bincount(ref_idx - min_idx)
    unique_matches = list(good_matches)
    for i, n in enumerate(num_matches):
        if n > 1:
            conflicts = (ref_idx == i + min_idx).nonzero()[0]
            for j in np.argsort(distances[conflicts])[1:]:
                unique_matches.remove(good_matches[conflicts[j]])

    # Extract coordinates and align based on matched sources
    return get_transform_stars(
        np.array([kp1[m.queryIdx].pt for m in unique_matches]) + 1,
        np.array([kp2[m.trainIdx].pt for m in unique_matches]) + 1,
        enable_rot=enable_rot,
        enable_scale=enable_scale,
        enable_skew=enable_skew,
    )


# noinspection PyPep8Naming
def get_transform_pixel(img: Union[np.ndarray, np.ma.MaskedArray],
                        ref_img: Union[np.ndarray, np.ma.MaskedArray],
                        enable_rot: bool = True,
                        enable_scale: bool = True,
                        enable_skew: bool = True,
                        detect_edges: bool = False) \
        -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Calculate the alignment transformation based on direct pixel-based
    registration

    :param img: input image as 2D NumPy array
    :param ref_img: reference image
    :param enable_rot: allow rotation transformation for >= 2 points
    :param enable_scale: allow scaling transformation for >= 2 points
    :param enable_skew: allow skew transformation for >= 2 points; ignored
        and set to False if `enable_rot`=False or `enable_scale`=False
    :param detect_edges: apply edge detection before gradient

    :return: 2x2 linear transformation matrix and offset vector [dy, dx]
    """
    ref_height, ref_width = ref_img.shape
    img, mask = match_ref_shape(img, ref_width, ref_height)[:2]
    img = img[:ref_height, :ref_width]
    mask = mask[:ref_height, :ref_width]
    if isinstance(ref_img, np.ma.MaskedArray) and ref_img.mask.any():
        mask |= ref_img.mask

    if detect_edges:
        img = np.hypot(
            nd.sobel(img, 0, mode='nearest'),
            nd.sobel(img, 1, mode='nearest')
        )
        ref_img = np.hypot(
            nd.sobel(ref_img, 0, mode='nearest'),
            nd.sobel(ref_img, 1, mode='nearest')
        )

    # The code below is ported from cv2.reg
    grady, gradx = np.gradient(ref_img)
    diff = ref_img - img
    diff[mask.nonzero()] = 0
    grid_r, grid_c = np.indices(img.shape) + 1

    if enable_rot and enable_scale and not enable_skew:
        # Similarity transform: rotation + uniform scale
        xIx_p_yIy = grid_c*gradx + grid_r*grady
        yIx_m_xIy = grid_r*gradx - grid_c*grady
        A = np.empty([4, 4], float)
        A[0, 0] = (xIx_p_yIy**2).sum()
        A[0, 1] = (xIx_p_yIy*yIx_m_xIy).sum()
        A[0, 2] = (gradx*xIx_p_yIy).sum()
        A[0, 3] = (grady*xIx_p_yIy).sum()
        A[1, 1] = (yIx_m_xIy**2).sum()
        A[1, 2] = (gradx*yIx_m_xIy).sum()
        A[1, 3] = (grady*yIx_m_xIy).sum()
        A[2, 2] = (gradx**2).sum()
        A[2, 3] = (gradx*grady).sum()
        A[3, 3] = (grady**2).sum()

        # Lower half values (A is symmetric)
        for i in range(1, 4):
            for j in range(i):
                A[i, j] = A[j, i]

        # Calculation of b
        b = [
            -(diff*xIx_p_yIy).sum(),
            -(diff*yIx_m_xIy).sum(),
            -(diff*gradx).sum(),
            -(diff*grady).sum(),
        ]

        # Calculate affine transformation
        k = np.dot(np.linalg.inv(A), b)
        return np.array([[k[0] + 1, k[1]], [-k[1], k[0] + 1]]), \
            np.array([k[2], k[3]])

    if enable_rot and not enable_scale:
        # Euclidean transform: rotation only
        xIy_yIx = grid_c*grady - grid_r*gradx
        A = np.empty([3, 3], float)
        A[0, 0] = (gradx**2).sum()
        A[0, 1] = A[1, 0] = (gradx*grady).sum()
        A[0, 2] = A[2, 0] = (gradx*xIy_yIx).sum()
        A[1, 1] = (grady**2).sum()
        A[1, 2] = A[2, 1] = (grady*xIy_yIx).sum()
        A[2, 2] = (xIy_yIx**2).sum()

        b = [
            -(diff*gradx).sum(),
            -(diff*grady).sum(),
            -(diff*xIy_yIx).sum(),
        ]

        # Calculate affine transformation
        k = np.dot(np.linalg.inv(A), b)
        c = np.cos(k[2])
        s = np.sin(k[2])
        return np.array([[c, -s], [s, c]]), np.array([k[0], k[1]])

    if not enable_rot:
        # Shift-only transform
        A = np.empty([2, 2], float)
        A[0, 0] = (gradx**2).sum()
        A[0, 1] = A[1, 0] = (gradx*grady).sum()
        A[1, 1] = (grady**2).sum()

        b = [-(diff*gradx).sum(), -(diff*grady).sum()]

        # Calculate affine transformation
        return None, np.dot(np.linalg.inv(A), b)

    # Full affine transform
    xIx = grid_c*gradx
    xIy = grid_c*grady
    yIx = grid_r*gradx
    yIy = grid_r*grady
    Ix2 = gradx*gradx
    Iy2 = grady*grady
    xy = grid_c*grid_r
    IxIy = gradx*grady
    A = np.empty([6, 6], float)
    A[0, 0] = (xIx**2).sum()
    A[0, 1] = (xy*Ix2).sum()
    A[0, 2] = (grid_c*Ix2).sum()
    A[0, 3] = (grid_c**2*IxIy).sum()
    A[0, 4] = A[1, 3] = (xy*IxIy).sum()
    A[0, 5] = A[2, 3] = (grid_c*IxIy).sum()
    A[1, 1] = (yIx**2).sum()
    A[1, 2] = (grid_r*Ix2).sum()
    A[1, 4] = (grid_r**2*IxIy).sum()
    A[1, 5] = A[2, 4] = (grid_r*IxIy).sum()
    A[2, 2] = Ix2.sum()
    A[2, 5] = IxIy.sum()
    A[3, 3] = (xIy**2).sum()
    A[3, 4] = (xy*Iy2).sum()
    A[3, 5] = (grid_c*Iy2).sum()
    A[4, 4] = (yIy**2).sum()
    A[4, 5] = (grid_r*Iy2).sum()
    A[5, 5] = Iy2.sum()
    # Lower half values (A is symmetric)
    for i in range(1, 6):
        for j in range(i):
            A[i, j] = A[j, i]

    # Calculation of b
    b = [
        -(diff*xIx).sum(),
        -(diff*yIx).sum(),
        -(diff*gradx).sum(),
        -(diff*xIy).sum(),
        -(diff*yIy).sum(),
        -(diff*grady).sum(),
    ]

    # Calculate affine transformation
    k = np.dot(np.linalg.inv(A), b)
    return (
        np.array([[k[0] + 1, k[1]], [k[3], k[4] + 1]]), np.array([k[2], k[5]]))


def apply_transform(img: Union[np.ndarray, np.ma.MaskedArray],
                    mat: Optional[np.ndarray],
                    offset: np.ndarray,
                    ref_width: int, ref_height: int,
                    prefilter: bool = True) -> np.ma.MaskedArray:
    """
    Apply alignment transform to the image

    :param img: input image as 2D NumPy array
    :param mat: optional 2x2 linear transformation matrix; None = unity matrix
        (offset only)
    :param offset: 2-element offset vector [dy, dx]
    :param ref_width: reference image width in pixels
    :param ref_height: reference image height in pixels
    :param prefilter: apply spline filter before interpolation

    :return: transformed image
    """
    # Pad the image if smaller than the reference image
    img, mask, avg = match_ref_shape(img, ref_width, ref_height)

    if mat is None:
        offset = -np.asarray(offset)
        img = nd.shift(img, offset, mode='nearest', prefilter=prefilter)
        mask = nd.shift(mask, offset, cval=1, prefilter=prefilter) > 0.06
    else:
        img = nd.affine_transform(
            img, mat, offset, mode='nearest', prefilter=prefilter)
        mask = nd.affine_transform(
            mask, mat, offset, cval=1, prefilter=prefilter) > 0.06

    # Match the reference image size
    return np.ma.masked_array(
        img[:ref_height, :ref_width], mask[:ref_height, :ref_width],
        fill_value=avg)
