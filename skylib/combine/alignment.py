"""
Image alignment.

:func:`~apply_transform_wcs()`: align an image based on WCS.
:func:`~apply_transform_stars()`: align an image based on pixel coordinates of
1, 2, or more stars.
"""

from __future__ import absolute_import, division, print_function

from numpy import (
    array, float32, indices, ma, mgrid, ones, sqrt, transpose, zeros)
from numpy.linalg import lstsq
import scipy.ndimage


__all__ = ['apply_transform_stars', 'apply_transform_wcs']


def apply_transform_stars(img, src_stars, dst_stars):
    """
    Align an image based on pixel coordinates of one or more stars

    :param array_like img: input image as 2D NumPy array
    :param list | array_like src_stars: list of (X, Y) coordinates of one
        or more alignment stars in the image being aligned
    :param list | array_like dst_stars: list of (X, Y) coordinates of the same
        stars as in `src_stars` in the reference image

    :return: transformed image
    :rtype: numpy.ma.MaskedArray
    """
    nref = min(len(src_stars), len(dst_stars))

    src_x, src_y = transpose(src_stars[:nref])
    dst_x, dst_y = transpose(dst_stars[:nref])

    avg = img.mean()
    if isinstance(img, ma.MaskedArray) and img.mask.any():
        # scipy.ndimage does not handle masked arrays; fill masked values with
        # global mean and mask them afterwards after transformation
        mask = img.mask.astype(float32)
        img = img.filled(avg)
    else:
        mask = zeros(img.shape)

    if nref == 1:
        # Pure shift
        offset = [dst_y[0] - src_y[0], dst_x[0] - src_x[0]]
        output = scipy.ndimage.shift(img, offset, mode='nearest')
        mask = scipy.ndimage.shift(mask, offset, cval=True)
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
        output = scipy.ndimage.affine_transform(
            img, mat, offset, mode='nearest')
        mask = scipy.ndimage.affine_transform(
            mask, mat, offset, cval=True) > 0.06

    return ma.MaskedArray(output, mask, fill_value=avg)


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


def apply_transform_wcs(img, src_wcs, dst_wcs, grid_points=0):
    """
    Align an image based on WCS

    :param array_like img: input image as 2D NumPy array
    :param astropy.wcs.WCS src_wcs: WCS of image being aligned
    :param astropy.wcs.WCS dst_wcs: reference image WCS
    :param int grid_points: number of grid points for WCS interpolation::
        0: transform using WCS calculated for each pixel
        1: offset-only alignment using central pixel
        2: shift + rotation + uniform scale (2-star) alignment using two points
        >= 3: full affine transform using the given number of fake "alignment
            stars" generated from the WCS

    :return: transformed image
    :rtype: numpy.ma.MaskedArray
    """
    h, w = img.shape
    if grid_points <= 0 or grid_points >= w*h:
        # Full geometric transform based on WCS
        avg = img.mean()
        if isinstance(img, ma.MaskedArray) and img.mask.any():
            mask = img.mask.astype(float32)
            img = img.filled(avg)
        else:
            mask = zeros(img.shape)

        dst_y, dst_x = indices(img.shape)
        a, d = dst_wcs.all_pix2world(dst_x, dst_y, 0)
        coordinates = src_wcs.all_world2pix(a, d, 0, quiet=True)[::-1]

        return ma.MaskedArray(
            scipy.ndimage.map_coordinates(img, coordinates, mode='nearest'),
            scipy.ndimage.map_coordinates(mask, coordinates, cval=1) > 0.06,
            fill_value=avg)

    # Calculate fake alignment stars by sampling WCS on a grid
    try:
        # Special grid for small number of points
        if w >= h:
            dst_x, dst_y = wcs_grid[grid_points]
        else:
            dst_y, dst_x = wcs_grid[grid_points]
        dst_x = dst_x*w  # cannot multiply in place to avoid damaging wcs_grid
        dst_y = dst_y*h
    except KeyError:
        # Generate uniform grid; not necessarily the requested number of points
        nx, ny = int(sqrt(grid_points*w/h) + 0.5), \
                 int(sqrt(grid_points*h/w) + 0.5)
        dst_x, dst_y = mgrid[:w:(nx + 2)*1j, :h:(ny + 2)*1j]
        dst_x, dst_y = dst_x[1:-1].ravel(), dst_y[1:-1].ravel()

    a, d = dst_wcs.all_pix2world(dst_x, dst_y, 0)
    src_x, src_y = src_wcs.all_world2pix(a, d, 0, quiet=True)

    return apply_transform_stars(
        img, transpose([src_x, src_y]), transpose([dst_x, dst_y]))
