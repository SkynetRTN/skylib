"""
Image alignment.

:func:`~apply_transform_wcs()`: align an image based on WCS.
:func:`~apply_transform_stars()`: align an image based on pixel coordinates of
1, 2, or more stars.
"""

from typing import List as TList, Tuple, Union

from numpy import (
    array, empty, float32, full, indices, ma, mgrid, ndarray, ones, sqrt,
    transpose, zeros)
from numpy.linalg import lstsq
import scipy.ndimage
from astropy.wcs import WCS


__all__ = ['apply_transform_stars', 'apply_transform_wcs']


def apply_transform_stars(img: ndarray,
                          src_stars: Union[TList[Tuple[float, float]],
                                           ndarray],
                          dst_stars: Union[TList[Tuple[float, float]],
                                           ndarray],
                          ref_width: int, ref_height: int,
                          prefilter: bool = True) -> ma.MaskedArray:
    """
    Align an image based on pixel coordinates of one or more stars

    :param img: input image as 2D NumPy array
    :param src_stars: list of (X, Y) coordinates of one or more alignment stars
        in the image being aligned
    :param dst_stars: list of (X, Y) coordinates of the same stars as in
        `src_stars` in the reference image
    :param ref_width: reference image width in pixels
    :param ref_height: reference image height in pixels
    :param prefilter: apply spline filter before interpolation

    :return: transformed image
    """
    nref = min(len(src_stars), len(dst_stars))

    src_x, src_y = transpose(src_stars[:nref])
    dst_x, dst_y = transpose(dst_stars[:nref])

    # Pad the image if smaller than the reference image
    h, w = img.shape
    avg = img.mean()
    if w < ref_width or h < ref_height:
        new_img = full([max(h, ref_height), max(w, ref_width)], avg, img.dtype)
        if isinstance(img, ma.MaskedArray) and img.mask.any():
            new_img[:h, :w] = img.data
            mask = ones([ref_height, ref_width], bool)
            mask[:h, :w] = img.mask
            img = ma.MaskedArray(new_img, mask)
        else:
            new_img[:h, :w] = img
            img = new_img

    if isinstance(img, ma.MaskedArray) and img.mask.any():
        # scipy.ndimage does not handle masked arrays; fill masked values with
        # global mean and mask them afterwards after transformation
        mask = img.mask.astype(float32)
        img = img.filled(avg)
    else:
        mask = zeros(img.shape, float32)

    if nref == 1:
        # Pure shift
        offset = [dst_y[0] - src_y[0], dst_x[0] - src_x[0]]
        img = scipy.ndimage.shift(
            img, offset, mode='nearest', prefilter=prefilter)
        mask = scipy.ndimage.shift(mask, offset, cval=True, prefilter=prefilter)
    else:
        if nref == 2:
            # Partial affine transform (shift + rotation + uniform scale)
            # [ src_y ]   [  A B ] [ dst_y ]   [ dy ]
            # [ src_x ] = [ -B A ] [ dst_x ] + [ dx ]
            src_dy, src_dx = src_y[0] - src_y[1], src_x[0] - src_x[1]
            dst_dy, dst_dx = dst_y[0] - dst_y[1], dst_x[0] - dst_x[1]
            d = dst_dx**2 + dst_dy**2
            if not d:
                raise ValueError(
                    'Both alignment stars have the same coordinates')
            a = (src_dy*dst_dy + src_dx*dst_dx)/d
            b = (src_dy*dst_dx - src_dx*dst_dy)/d
            mat = array([[a, b], [-b, a]])
            offset = [src_y[0] - dst_y[0]*a - dst_x[0]*b,
                      src_x[0] - dst_x[0]*a + dst_y[0]*b]
        else:
            # Full affine transform
            # [ src_y ]   [ A B ] [ dst_y ]   [ dy ]
            # [ src_x ] = [ C D ] [ dst_x ] + [ dx ]
            a = transpose([dst_y, dst_x, ones(nref)])
            py = lstsq(a, src_y, rcond=None)[0]
            px = lstsq(a, src_x, rcond=None)[0]
            mat = array([py[:2], px[:2]])
            offset = [py[2], px[2]]
        img = scipy.ndimage.affine_transform(
            img, mat, offset, mode='nearest', prefilter=prefilter)
        mask = scipy.ndimage.affine_transform(
            mask, mat, offset, cval=True, prefilter=prefilter) > 0.06

    # Match the reference image size
    if w > ref_width or h > ref_height:
        img = img[:ref_height, :ref_width]
        mask = mask[:ref_height, :ref_width]

    return ma.masked_array(img, mask, fill_value=avg)


wcs_grid = {
    1: (array([1/2]),
        array([1/2])),
    2: (array([1/3, 2/3]),
        array([1/2, 1/2])),
    3: (array([1/4, 1/2, 3/4]),
        array([1/3, 2/3, 1/3])),
    4: (array([1/3, 2/3, 1/3, 2/3]),
        array([1/3, 1/3, 2/3, 2/3])),
    5: (array([1/3, 2/3, 1/3, 2/3, 1/2]),
        array([1/3, 1/3, 2/3, 2/3, 1/2])),
    6: (array([1/4, 1/2, 3/4, 1/4, 1/2, 3/4]),
        array([1/3, 1/3, 1/3, 2/3, 2/3, 2/3])),
    7: (array([1/4, 1/2, 3/4, 1/4, 1/2, 3/4, 1/2]),
        array([1/3, 1/3, 1/3, 2/3, 2/3, 2/3, 1/2])),
    8: (array([1/4, 1/2, 3/4, 1/4, 1/2, 3/4, 1/3, 2/3]),
        array([1/3, 1/3, 1/3, 2/3, 2/3, 2/3, 1/2, 1/2])),
    9: (array([1/4, 1/2, 3/4, 1/4, 1/2, 3/4, 1/4, 1/2, 3/4]),
        array([1/4, 1/4, 1/4, 1/2, 1/2, 1/2, 3/4, 3/4, 3/4])),
}


def apply_transform_wcs(img: ndarray, src_wcs: WCS, dst_wcs: WCS,
                        ref_width: int, ref_height: int,
                        grid_points: int = 0,
                        prefilter: bool = False) -> ma.MaskedArray:
    """
    Align an image based on WCS

    :param img: input image as 2D NumPy array
    :param src_wcs: WCS of image being aligned
    :param dst_wcs: reference image WCS
    :param ref_width: reference image width in pixels
    :param ref_height: reference image height in pixels
    :param grid_points: number of grid points for WCS interpolation::
        0: transform using WCS calculated for each pixel
        1: offset-only alignment using central pixel
        2: shift + rotation + uniform scale (2-star) alignment using two points
        >= 3: full affine transform using the given number of fake "alignment
            stars" generated from the WCS
    :param prefilter: apply spline filter before interpolation

    :return: transformed image
    """
    # Pad the image if smaller than the reference image
    h, w = img.shape
    avg = img.mean()
    if w < ref_width or h < ref_height:
        new_img = full([max(h, ref_height), max(w, ref_width)], avg, img.dtype)
        if isinstance(img, ma.MaskedArray) and img.mask.any():
            new_img[:h, :w] = img.data
            mask = ones([ref_height, ref_width], bool)
            mask[:h, :w] = img.mask
            img = ma.MaskedArray(new_img, mask)
        else:
            new_img[:h, :w] = img
            img = new_img

    if grid_points <= 0 or grid_points >= w*h:
        # Full geometric transform based on WCS
        if isinstance(img, ma.MaskedArray) and img.mask.any():
            mask = img.mask.astype(float32)
            img = img.filled(avg)
        else:
            mask = zeros(img.shape, float32)

        # Calculate the transformation row by row to avoid problems
        # in all_pix2world() for large images
        dst_y, dst_x = indices((ref_height, ref_width))
        coord = empty((2, h, w), float32)
        for i in range(h):
            a, d = dst_wcs.all_pix2world(dst_x[i], dst_y[i], 0)
            coord[1, i, :], coord[0, i, :] = src_wcs.all_world2pix(
                a, d, 0, quiet=True)

        res = ma.MaskedArray(
            scipy.ndimage.map_coordinates(
                img, coord, mode='nearest', prefilter=prefilter),
            scipy.ndimage.map_coordinates(
                mask, coord, cval=1, prefilter=prefilter) > 0.06,
            fill_value=avg)

        # Match the reference image size
        if w > ref_width or h > ref_height:
            res = res[:ref_height, :ref_width]

        return res

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
        nx, ny = int(sqrt(grid_points*ref_width/ref_height) + 0.5), \
                 int(sqrt(grid_points*ref_height/ref_width) + 0.5)
        dst_x, dst_y = mgrid[:ref_width:(nx + 2)*1j, :ref_height:(ny + 2)*1j]
        dst_x, dst_y = dst_x[1:-1].ravel(), dst_y[1:-1].ravel()

    a, d = dst_wcs.all_pix2world(dst_x, dst_y, 0)
    src_x, src_y = src_wcs.all_world2pix(a, d, 0, quiet=True)

    img = apply_transform_stars(
        img, transpose([src_x, src_y]), transpose([dst_x, dst_y]),
        ref_width, ref_height, prefilter=prefilter)

    # Match the reference image size
    if w > ref_width or h > ref_height:
        img = img[:ref_height, :ref_width]

    return img
