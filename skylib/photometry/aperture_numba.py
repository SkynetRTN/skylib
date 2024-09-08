"""
Parallel Numba implementation of aperture photometry based on SEP source code with optional background outlier
rejection.

Uses exact sub-pixel math (subpix = 0).

Note the difference with SEP in sum_*() outputs: (flux, fluxerr, area, flags) instead of (flux, fluxerr, flags).
"""

import numpy as np
from numba import prange

from ..util.overlap import circoverlap, ellipoverlap, njitc
from ..util.stats import chauvenet1


__all__ = ['sum_circle', 'sum_circann', 'sum_ellipse', 'sum_ellipann']


# Aperture flags
SEP_APER_TRUNC = 0x0010
SEP_APER_HASMASKED = 0x0020


@njitc
def ellipse_coeffs(a: float, b: float, theta: float) -> tuple[float, float, float]:
    """Convert ellipse parameters (a, b, theta) into coeffs (cxx, cyy, cxy)"""
    ctheta = np.cos(theta)
    stheta = np.sin(theta)
    return ctheta**2/a**2 + stheta**2/b**2, stheta**2/a**2 + ctheta**2/b**2, 2*ctheta*stheta*(1/a**2 - 1/b**2)


@njitc
def boxextent(x: float, y: float, rx: float, ry: float, w: int, h: int) -> tuple[int, int, int, int, int]:
    """
    Determine the extent of the box enclosing axis-aligned ellipse with semi-axes (rx, ry) centered at (x, y).

    :param x: aperture center X
    :param y: aperture center Y
    :param rx: aperture semi-major axis
    :param ry: aperture semi-minor axis
    :param w: image width
    :param h: image height

    :return: xmin, xmax, ymin, ymax, flag

    xmin, ymin are inclusive and xmax, ymax are exclusive.
    Ensures that box is within image bound and sets a flag if it is not.
    """
    flag = 0
    xmin = int(x - rx + 0.5)
    if xmin < 0:
        xmin = 0
        flag |= SEP_APER_TRUNC
    xmax = int(x + rx + 1.4999999)
    if xmax > w:
        xmax = w
        flag |= SEP_APER_TRUNC
    ymin = int(y - ry + 0.5)
    if ymin < 0:
        ymin = 0
        flag |= SEP_APER_TRUNC
    ymax = int(y + ry + 1.4999999)
    if ymax > h:
        ymax = h
        flag |= SEP_APER_TRUNC
    return xmin, xmax, ymin, ymax, flag


@njitc
def boxextent_ellipse(x: float, y: float, cxx: float, cyy: float, cxy: float, r: float, w: int, h: int) \
        -> tuple[int, int, int, int, int]:
    """
    Determine the extent of the box enclosing ellipse

    :param x: aperture center X
    :param y: aperture center Y
    :param cxx: ellipse parameter (see :func:`ellipse_coeffs`)
    :param cyy: --//--
    :param cxy: --//--
    :param r: aperture size scaling factor
    :param w: image width
    :param h: image height

    :return: xmin, xmax, ymin, ymax, flag
    """
    dxlim = cxx - cxy**2/(4*cyy)
    dxlim = r/np.sqrt(dxlim) if dxlim > 0 else 0
    dylim = cyy - cxy**2/(4*cxx)
    dylim = r/np.sqrt(dylim) if dylim > 0 else 0
    return boxextent(x, y, dxlim, dylim, w, h)


@njitc
def oversamp_ann_circle(r: float) -> tuple[float, float]:
    """determine oversampled "annulus" for a circle"""
    r_in2 = r - 0.7072
    return r_in2**2 if r_in2 > 0 else 0, (r + 0.7072)**2


@njitc
def oversamp_ann_ellipse(a: float, b: float) -> tuple[float, float]:
    """determine oversampled "annulus" for an ellipse"""
    r_in2 = a - 0.7072/b
    return r_in2**2 if r_in2 > 0 else 0, (a + 0.7072/b)**2


def sum_aper_factory(aper_init,
                     aper_boxextent,
                     aper_rpix2,
                     aper_compare1,
                     aper_compare2,
                     aper_exact) -> tuple:
    """
    Create a pair of jitted functions that sum data over a specific aperture shape, one without and one with outlier
    rejection

    :param aper_init: function that checks the input aperture parameters, raises ValueError if they are invalid, and
        returns 1D array of internal aperture parameters used by the functions below::
            def aper_init(aper: np.ndarray) -> np.ndarray:
                ...
        `aper` is 1D array of aperture-specific parameters
    :param aper_boxextent: function that returns the extent of the box enclosing the aperture::
            def aper_boxextent(aper: np.ndarray) -> np.ndarray:
                ...
    :param aper_rpix2: function that returns normalized squared distance from the aperture center::
            def aper_rpix2(dx: float, dy: float, aper_params: np.ndarray) -> float:
                ...
        `dx` and `dy` are pixel coordinates relative to the aperture center
        `aper_params` is array of internal aperture-specific parameters returned by `aper_init`
    :param aper_compare1: function that returns True if the given pixel is at least partially within the aperture::
            def aper_compare1(rpix2: float, aper_params: np.ndarray) -> bool:
                ...
        `rpix2` is the value returned by `aper_rpix2`
    :param aper_compare2: function that returns True if the given pixel is not fully within the aperture::
            def aper_compare2(rpix2: float, aper_params: np.ndarray) -> bool:
                ...
    :param aper_exact: function that returns the amount of overlap of the given pixel with the aperture::
            def aper_exact(dx: float, dy: float, aper_params: np.ndarray) -> float:
                ...

    :return: two summation function instances for the given aperture shape
    """

    @njitc(cache=False)
    def _sum_aper(x: float,
                  y: float,
                  aper: np.ndarray,
                  data: np.ndarray,
                  mask: np.ndarray | None,
                  maskthresh: float,
                  noise: np.ndarray,
                  segmap: np.ndarray | None,
                  seg_id: int,
                  gain: float,
                  ignore_mask: bool = False) -> tuple[float, float, float, int]:
        """
        Sum pixels over the given aperture with subpixel accuracy

        :param x: aperture center X (0-based)
        :param y: aperture center Y (0-based)
        :param aper: 1D array of aperture-specific parameters
        :param data: 2D image data array
        :param mask: optional 2D mask array, same shape as `data`
        :param maskthresh: consider pixel masked if `mask`[i, j] > `maskthresh`
        :param noise: 2D array of image noise, same shape as `data` or a single-element 1x1 array if noise is constant
        :param segmap: optional segmentation map, same shape as `data`; sources are indicated by segmap[i, j] > 0,
            background corresponds to segmap[i, j] = 0
        :param seg_id: source ID in the segmentation map::
            * seg_id > 0: ignore all non-background pixels except those with segmap[i, j] == seg_id
            * seg_id < 0: ignore all (including background) pixels except those with segmap[i, j] == -seg_id
        :param gain: inverse camera gain in e-/count
        :param ignore_mask: if True, exclude masked pixels from the total aperture area; otherwise, rescale flux and its
            error to the total aperture area, including masked pixels

        :return::
            * total flux over the aperture
            * flux error
            * total aperture area
            * aperture flags (SEP_APER_TRUNC and/or SEP_APER_HASMASKED)
        """
        # initializations
        tv = sigtv = totarea = maskarea = 0.0
        h, w = data.shape

        # Initialize aperture-specific parameters
        aper_params = aper_init(aper)

        # Scalar noise?
        if noise.size == 1:
            noiseval = noise[0, 0]
        else:
            noiseval = 0

        # get extent of box
        xmin, xmax, ymin, ymax, flag = aper_boxextent(x, y, w, h, aper_params)

        # loop over rows in the box
        for iy in range(ymin, ymax):
            # loop over pixels in this row
            for ix in range(xmin, xmax):
                dx = ix - x
                dy = iy - y
                rpix2 = aper_rpix2(dx, dy, aper_params)
                if aper_compare1(rpix2, aper_params):
                    if aper_compare2(rpix2, aper_params):  # might be partially in aperture
                        overlap = min(max(aper_exact(dx, dy, aper_params), 0), 1)
                    else:
                        # definitely fully in aperture
                        overlap = 1.0

                    ismasked = mask is not None and mask[iy, ix] > maskthresh

                    # Segmentation image:
                    #    If `id` is negative, require segmented pixels within the aperture.
                    #    If `id` is positive, mask pixels with nonzero segment ids not equal to `id`.
                    if not ismasked and segmap is not None:
                        segt = segmap[iy, ix]
                        if seg_id > 0:
                            if segt > 0 and segt != seg_id:
                                ismasked = True
                        elif segt != -seg_id:
                            ismasked = True

                    if ismasked:
                        flag |= SEP_APER_HASMASKED
                        maskarea += overlap
                    else:
                        v = data[iy, ix]
                        tv += v*overlap
                        if noise.size > 1:
                            noiseval = noise[iy, ix]
                        if noiseval:
                            sigtv += noiseval*overlap

                    totarea += overlap

        # correct for masked values
        if maskarea > 0:
            if ignore_mask:
                totarea -= maskarea
            elif totarea > maskarea:
                tmp = totarea/(totarea - maskarea)
                tv *= tmp
                sigtv *= tmp

        # add poisson noise, only if gain > 0
        if gain > 0 and tv > 0:
            sigtv += tv/gain

        return tv, np.sqrt(sigtv), totarea, flag

    @njitc(cache=False)
    def _sum_aper_reject(x: float,
                         y: float,
                         aper: np.ndarray,
                         data: np.ndarray,
                         mask: np.ndarray | None,
                         maskthresh: float,
                         noise: np.ndarray,
                         segmap: np.ndarray | None,
                         seg_id: int,
                         gain: float,
                         ignore_mask: bool = False) -> tuple[float, float, float, int]:
        """
        Sum pixels over the given aperture with subpixel accuracy and outlier rejection

        Inputs and outputs -- see :func:`_sum_aper`
        """
        # initializations
        h, w = data.shape

        # Initialize aperture-specific parameters
        aper_params = aper_init(aper)

        # Scalar noise?
        if noise.size == 1:
            noiseval = noise[0, 0]
        else:
            noiseval = 0

        # get extent of box
        xmin, xmax, ymin, ymax, flag = aper_boxextent(x, y, w, h, aper_params)

        nmax = (xmax - xmin)*(ymax - ymin)
        a = np.empty((nmax, 3), np.float64)
        b = np.empty(nmax, np.float64)
        overlap = np.empty(nmax, np.float64)

        # Extract non-masked pixels within the aperture
        npix = 0
        totarea = maskarea = 0.0
        for iy in range(ymin, ymax):
            for ix in range(xmin, xmax):
                dx = ix - x
                dy = iy - y
                rpix2 = aper_rpix2(dx, dy, aper_params)
                if aper_compare1(rpix2, aper_params):
                    if aper_compare2(rpix2, aper_params):  # might be partially in aperture
                        ov = overlap[npix] = min(max(aper_exact(dx, dy, aper_params), 0), 1)
                    else:
                        # definitely fully in aperture
                        ov = overlap[npix] = 1.0
                    if not ov:
                        continue

                    ismasked = mask is not None and mask[iy - ymin, ix - xmin] > maskthresh

                    # Segmentation image:
                    #    If `id` is negative, require segmented pixels within the aperture.
                    #    If `id` is positive, mask pixels with nonzero segment ids not equal to `id`.
                    if not ismasked and segmap is not None:
                        segt = segmap[iy, ix]
                        if seg_id > 0:
                            if segt > 0 and segt != seg_id:
                                ismasked = True
                        elif segt != -seg_id:
                            ismasked = True

                    if ismasked:
                        flag |= SEP_APER_HASMASKED
                        maskarea += ov
                    else:
                        a[npix, 0] = ix
                        a[npix, 1] = iy
                        a[npix, 2] = 1
                        b[npix] = data[iy, ix]
                        npix += 1

                    totarea += ov

        if not npix:
            return 0.0, 0.0, 0.0, flag

        while True:
            tv = sigtv = 0.0
            for i in range(npix):
                ov = overlap[i]
                tv += b[i]*ov
                if noise.size > 1:
                    ix, iy = int(a[i, 0]), int(a[i, 1])
                    noiseval = noise[iy, ix]
                if noiseval:
                    sigtv += noiseval*ov

            p = np.linalg.lstsq(a[:npix], b[:npix])[0]
            resid = p[0]*a[:npix, 0] + p[1]*a[:npix, 1] + p[2] - b[:npix]
            bad = chauvenet1(
                resid, np.zeros(resid.shape, np.bool_), 0, 10, 1, None, 1, None, True, True, 0, None, 0.683)[0]
            if not bad.any():
                break

            i = 0
            while i < npix:
                if bad[i]:
                    maskarea += overlap[i]
                    a[i:npix - 1] = a[i + 1:npix]
                    b[i:npix - 1] = b[i + 1:npix]
                    overlap[i:npix - 1] = overlap[i + 1:npix]
                    bad[i:npix - 1] = bad[i + 1:npix]
                    npix -= 1
                    flag |= SEP_APER_HASMASKED
                else:
                    i += 1
            if not npix:
                break

        # correct for masked values
        if maskarea > 0:
            if ignore_mask:
                totarea -= maskarea
            elif totarea > maskarea:
                tmp = totarea/(totarea - maskarea)
                tv *= tmp
                sigtv *= tmp

        # add poisson noise, only if gain > 0
        if gain > 0 and tv > 0:
            sigtv += tv/gain

        return tv, np.sqrt(sigtv), totarea, flag

    return _sum_aper, _sum_aper_reject


@njitc(inline='always')
def _aper_init_circle(aper: np.ndarray) -> np.ndarray:
    if aper[0] < 0:
        raise ValueError('Negative aperture radius')
    aper_params = np.empty(3, np.float64)
    r = float(aper[0])
    aper_params[0] = r
    aper_params[1], aper_params[2] = oversamp_ann_circle(r)
    return aper_params


@njitc(inline='always')
def _aper_boxextent_circle(x: float, y: float, w: int, h: int, aper_params: np.ndarray) \
        -> tuple[int, int, int, int, int]:
    r = float(aper_params[0])
    return boxextent(x, y, r, r, w, h)


@njitc(inline='always')
def _aper_rpix2_circle(dx: float, dy: float, _aper_params: np.ndarray) -> float:
    return dx**2 + dy**2


@njitc(inline='always')
def _aper_compare1_circle(rpix2: float, aper_params: np.ndarray) -> bool:
    return rpix2 <= float(aper_params[2])


@njitc(inline='always')
def _aper_compare2_circle(rpix2: float, aper_params: np.ndarray) -> bool:
    return rpix2 >= float(aper_params[1])


@njitc(inline='always')
def _aper_exact_circle(dx: float, dy: float, aper_params: np.ndarray) -> float:
    return circoverlap(dx - 0.5, dy - 0.5, dx + 0.5, dy + 0.5, float(aper_params[0]))


_sum_circle, _sum_circle_reject = sum_aper_factory(
    _aper_init_circle, _aper_boxextent_circle, _aper_rpix2_circle, _aper_compare1_circle, _aper_compare2_circle,
    _aper_exact_circle)


@njitc(inline='always')
def _aper_init_ellipse(aper: np.ndarray) -> np.ndarray:
    if aper[0] < 0:
        raise ValueError('Negative aperture semi-major axis')
    if aper[1] < 0:
        raise ValueError('Negative aperture semi-minor axis')
    if aper[0] < aper[1]:
        raise ValueError('Aperture semi-major axis smaller than semi-minor axis')

    aper_params = np.empty(9, np.float64)
    aper_params[:4] = a, b, theta, r = aper
    aper_params[:2] *= r
    aper_params[4:6] = oversamp_ann_ellipse(r, b)
    aper_params[6:9] = ellipse_coeffs(a, b, theta)
    return aper_params


@njitc(inline='always')
def _aper_boxextent_ellipse(x: float, y: float, w: int, h: int, aper_params: np.ndarray) \
        -> tuple[int, int, int, int, int]:
    return boxextent_ellipse(
        x, y, float(aper_params[6]), float(aper_params[7]), float(aper_params[8]), float(aper_params[3]), w, h)


@njitc(inline='always')
def _aper_rpix2_ellipse(dx: float, dy: float, aper_params: np.ndarray) -> float:
    return float(aper_params[6])*dx**2 + float(aper_params[7])*dy**2 + float(aper_params[8])*dx*dy


@njitc(inline='always')
def _aper_compare1_ellipse(rpix2: float, aper_params: np.ndarray) -> bool:
    return rpix2 <= float(aper_params[5])


@njitc(inline='always')
def _aper_compare2_ellipse(rpix2: float, aper_params: np.ndarray) -> bool:
    return rpix2 >= float(aper_params[4])


@njitc(inline='always')
def _aper_exact_ellipse(dx: float, dy: float, aper_params: np.ndarray) -> float:
    return ellipoverlap(
        dx - 0.5, dy - 0.5, dx + 0.5, dy + 0.5, float(aper_params[0]), float(aper_params[1]), float(aper_params[2]))


_sum_ellipse, _sum_ellipse_reject = sum_aper_factory(
    _aper_init_ellipse, _aper_boxextent_ellipse, _aper_rpix2_ellipse, _aper_compare1_ellipse, _aper_compare2_ellipse,
    _aper_exact_ellipse)


@njitc(inline='always')
def _aper_init_circann(aper: np.ndarray) -> np.ndarray:
    if aper[0] < 0:
        raise ValueError('Negative inner annulus radius')
    if aper[0] > aper[1]:
        raise ValueError('Inner annulus radius must be smaller than outer annulus radius')

    aper_params = np.empty(6, np.float64)
    aper_params[:2] = r_in, r_out = aper
    aper_params[2:4] = oversamp_ann_circle(r_in)
    aper_params[4:6] = oversamp_ann_circle(r_out)
    return aper_params


@njitc(inline='always')
def _aper_boxextent_circann(x: float, y: float, w: int, h: int, aper_params: np.ndarray) \
        -> tuple[int, int, int, int, int]:
    r_out = float(aper_params[1])
    return boxextent(x, y, r_out, r_out, w, h)


@njitc(inline='always')
def _aper_compare1_circann(rpix2: float, aper_params: np.ndarray) -> bool:
    return float(aper_params[2]) <= rpix2 <= float(aper_params[5])


@njitc(inline='always')
def _aper_compare2_circann(rpix2: float, aper_params: np.ndarray) -> bool:
    return rpix2 >= float(aper_params[4]) or rpix2 <= float(aper_params[3])


@njitc(inline='always')
def _aper_exact_circann(dx: float, dy: float, aper_params: np.ndarray) -> float:
    rout = float(aper_params[1])
    if rout <= 0:
        return 0
    overlap = circoverlap(dx - 0.5, dy - 0.5, dx + 0.5, dy + 0.5, rout)
    rin = float(aper_params[0])
    if rin > 0:
        overlap -= circoverlap(dx - 0.5, dy - 0.5, dx + 0.5, dy + 0.5, rin)
    return overlap


_sum_circann, _sum_circann_reject = sum_aper_factory(
    _aper_init_circann, _aper_boxextent_circann, _aper_rpix2_circle, _aper_compare1_circann, _aper_compare2_circann,
    _aper_exact_circann)


@njitc(inline='always')
def _aper_init_ellipann(aper: np.ndarray) -> np.ndarray:
    if aper[0] < 0:
        raise ValueError('Negative aperture semi-major axis')
    if aper[1] < 0:
        raise ValueError('Negative aperture semi-minor axis')
    if aper[0] < aper[1]:
        raise ValueError('Aperture semi-major axis smaller than semi-minor axis')
    if aper[3] < 0:
        raise ValueError('Negative inner annulus radius')
    if aper[3] > aper[3]:
        raise ValueError('Inner annulus radius must be smaller than outer annulus radius')

    aper_params = np.empty(12, np.float64)
    aper_params[:5] = a, b, theta, rin, rout = aper
    aper_params[5:7] = oversamp_ann_ellipse(rin, b)
    aper_params[7:9] = oversamp_ann_ellipse(rout, b)
    aper_params[9:12] = ellipse_coeffs(a, b, theta)
    return aper_params


@njitc(inline='always')
def _aper_boxextent_ellipann(x: float, y: float, w: int, h: int, aper_params: np.ndarray) \
        -> tuple[int, int, int, int, int]:
    return boxextent_ellipse(
        x, y, float(aper_params[9]), float(aper_params[10]), float(aper_params[11]), float(aper_params[4]), w, h)


@njitc(inline='always')
def _aper_rpix2_ellipann(dx: float, dy: float, aper_params: np.ndarray) -> float:
    return float(aper_params[9])*dx**2 + float(aper_params[10])*dy**2 + float(aper_params[11])*dx*dy


@njitc(inline='always')
def _aper_compare1_ellipann(rpix2: float, aper_params: np.ndarray) -> bool:
    return float(aper_params[5]) <= rpix2 <= float(aper_params[8])


@njitc(inline='always')
def _aper_compare2_ellipann(rpix2: float, aper_params: np.ndarray) -> bool:
    return rpix2 >= float(aper_params[7]) or rpix2 <= float(aper_params[6])


@njitc(inline='always')
def _aper_exact_ellipann(dx: float, dy: float, aper_params: np.ndarray) -> float:
    rout = float(aper_params[4])
    if rout <= 0:
        return 0
    a = float(aper_params[0])
    b = float(aper_params[1])
    theta = float(aper_params[2])
    rin = float(aper_params[3])
    overlap = ellipoverlap(dx - 0.5, dy - 0.5, dx + 0.5, dy + 0.5, a*rout, b*rout, theta)
    if rin > 0:
        overlap -= ellipoverlap(dx - 0.5, dy - 0.5, dx + 0.5, dy + 0.5, a*rin, b*rin, theta)
    return overlap


_sum_ellipann, _sum_ellipann_reject = sum_aper_factory(
    _aper_init_ellipann, _aper_boxextent_ellipann, _aper_rpix2_ellipann, _aper_compare1_ellipann,
    _aper_compare2_ellipann, _aper_exact_ellipann)


@njitc
def _prepare_arrays(
        data: np.ndarray,
        x: float | np.ndarray,
        y: float | np.ndarray,
        var: float | np.ndarray = 0,
        err: float | np.ndarray = 0,
        mask: np.ndarray | None = None,
        seg_id: int | np.ndarray | None = None,
        segmap: np.ndarray | None = None) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Helper function for sum_*() used to check and initialize the arrays common to all these functions
    """
    data_shape = data.shape

    # Check if noise is error or variance.
    err = np.asarray(err)
    var = np.asarray(var)
    if err.size > 1 or err.ravel()[0]:
        if var.size > 1 or var.ravel()[0]:
            raise ValueError('Cannot specify both err and var')
        if err.size == 1:
            # Scalar stddev
            noise = np.full((1, 1), err.ravel()[0]**2, np.float64)
        elif err.ndim == 2:
            # Array stddev
            if err.shape != data_shape:
                raise ValueError('Size of error array must match data')
            noise = np.empty(data.shape)
            noise[:] = err**2
        else:
            raise ValueError('Error array must be 0-d or 2-d')
    elif var.size > 1 or var.ravel()[0]:
        if var.size == 1:
            # Scalar variance
            noise = np.full((1, 1), var.ravel()[0], np.float64)
        elif var.ndim == 2:
            # Array variance
            if var.shape != data_shape:
                raise ValueError('Size of variance array must match data')
            noise = np.empty(data.shape)
            noise[:] = var
        else:
            raise ValueError('Variance array must be 0-d or 2-d')
    else:
        noise = np.zeros((1, 1), np.float64)

    # Optional input: mask
    if mask is not None and mask.shape != data_shape:
        raise ValueError('Size of mask array must match data')

    x = np.asarray(x).ravel()
    n = x.size
    y = np.asarray(y).ravel()
    if y.size != n:
        raise ValueError('Size of `y` array must match `x`')

    # Optional input: segmap
    if segmap is not None:
        if segmap.shape != data_shape:
            raise ValueError('Size of segmap array must match data')
        # Test for map without seg_id.  Nothing happens if seg_id supplied but without segmap.
        if seg_id is None:
            raise ValueError('`segmap` supplied but not `seg_id`')
        seg_id = np.asarray(seg_id).ravel()
        if seg_id.size == 1 and n != 1:
            seg_id = np.repeat(seg_id[0], n)
        elif seg_id.size != n:
            raise ValueError('Shapes of `x` and `seg_id` do not match')
    else:
        seg_id = np.zeros(n, np.uint8)

    sumdata = np.empty_like(x, np.float64)
    sumerr = np.empty_like(x, np.float64)
    area = np.empty_like(x, np.float64)
    flag = np.empty_like(x, np.int16)

    return x, y, noise, seg_id, sumdata, sumerr, area, flag


@njitc(parallel=True)
def sum_circle(
        data: np.ndarray,
        x: float | np.ndarray,
        y: float | np.ndarray,
        r: float | np.ndarray,
        var: float | np.ndarray = 0,
        err: float | np.ndarray = 0,
        gain: float = 0,
        mask: np.ndarray | None = None,
        maskthresh: float = 0,
        seg_id: int | np.ndarray | None = None,
        segmap: np.ndarray | None = None,
        bkgann: tuple[float | np.ndarray, float | np.ndarray] | None = None,
        reject_outliers: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parallel Numba port of :func:`sep.sum_circle`; always assumes subpix=0
    """
    x, y, noise, seg_id, sumdata, sumerr, area, flag = _prepare_arrays(data, x, y, var, err, mask, seg_id, segmap)
    n = x.size

    # Broadcast aperture parameters to x.shape
    aper = np.empty((n, 1), np.float64)
    aper[:, 0] = r

    if bkgann is None:
        if reject_outliers:
            for i in prange(n):
                sumdata[i], sumerr[i], area[i], flag[i] = _sum_circle_reject(
                    x[i], y[i], aper[i], data, mask, maskthresh, noise, segmap, seg_id[i], gain)
        else:
            for i in prange(n):
                sumdata[i], sumerr[i], area[i], flag[i] = _sum_circle(
                    x[i], y[i], aper[i], data, mask, maskthresh, noise, segmap, seg_id[i], gain)
    else:
        aper_ann = np.empty((n, 2), np.float64)
        aper_ann[:, 0] = bkgann[0]
        aper_ann[:, 1] = bkgann[1]

        for i in prange(n):
            flux, fluxerr, area[i], flag[i] = _sum_circle(
                x[i], y[i], aper[i], data, mask, maskthresh, noise, segmap, seg_id[i], gain)
            if reject_outliers:
                bkgflux, bkgfluxerr, bkgarea, _ = _sum_circann_reject(
                    x[i], y[i], aper_ann[i], data, mask, maskthresh, noise, segmap, seg_id[i], gain, ignore_mask=True)
            else:
                bkgflux, bkgfluxerr, bkgarea, _ = _sum_circann(
                    x[i], y[i], aper_ann[i], data, mask, maskthresh, noise, segmap, seg_id[i], gain, ignore_mask=True)
            _area = area[i]
            if _area > 0 and bkgarea > 0:
                flux -= bkgflux/bkgarea*_area
                fluxerr = np.hypot(fluxerr, bkgfluxerr*_area/bkgarea)
            sumdata[i], sumerr[i] = flux, fluxerr

    return sumdata, sumerr, area, flag


@njitc(parallel=True)
def sum_circann(
        data: np.ndarray,
        x: float | np.ndarray,
        y: float | np.ndarray,
        rin: float | np.ndarray,
        rout: float | np.ndarray,
        var: float | np.ndarray = 0,
        err: float | np.ndarray = 0,
        gain: float = 0,
        mask: np.ndarray | None = None,
        maskthresh: float = 0,
        seg_id: int | np.ndarray | None = None,
        segmap: np.ndarray | None = None,
        reject_outliers: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parallel Numba port of :func:`sep.sum_circann`; always assumes subpix=0
    """
    x, y, noise, seg_id, sumdata, sumerr, area, flag = _prepare_arrays(data, x, y, var, err, mask, seg_id, segmap)
    n = x.size

    # Broadcast aperture parameters to x.shape
    aper = np.empty((n, 2), np.float64)
    aper[:, 0] = rin
    aper[:, 1] = rout

    if reject_outliers:
        for i in prange(x.size):
            sumdata[i], sumerr[i], area[i], flag[i] = _sum_circann_reject(
                x[i], y[i], aper[i], data, mask, maskthresh, noise, segmap, seg_id[i], gain)
    else:
        for i in prange(x.size):
            sumdata[i], sumerr[i], area[i], flag[i] = _sum_circann(
                x[i], y[i], aper[i], data, mask, maskthresh, noise, segmap, seg_id[i], gain)

    return sumdata, sumerr, area, flag


@njitc(parallel=True)
def sum_ellipse(
        data: np.ndarray,
        x: float | np.ndarray,
        y: float | np.ndarray,
        a: float | np.ndarray,
        b: float | np.ndarray,
        theta: float | np.ndarray,
        r: float | np.ndarray = 1,
        var: float | np.ndarray = 0,
        err: float | np.ndarray = 0,
        gain: float = 0,
        mask: np.ndarray | None = None,
        maskthresh: float = 0,
        seg_id: int | np.ndarray | None = None,
        segmap: np.ndarray | None = None,
        bkgann: tuple[float | np.ndarray, float | np.ndarray] | None = None,
        reject_outliers: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parallel Numba port of :func:`sep.sum_ellipse`; always assumes subpix=0
    """
    x, y, noise, seg_id, sumdata, sumerr, area, flag = _prepare_arrays(data, x, y, var, err, mask, seg_id, segmap)
    n = x.size

    # Broadcast aperture parameters to x.shape
    aper = np.empty((n, 4), np.float64)
    aper[:, 0] = a
    aper[:, 1] = b
    aper[:, 2] = theta
    aper[:, 3] = r

    if bkgann is None:
        if reject_outliers:
            for i in prange(n):
                sumdata[i], sumerr[i], area[i], flag[i] = _sum_ellipse_reject(
                    x[i], y[i], aper[i], data, mask, maskthresh, noise, segmap, seg_id[i], gain)
        else:
            for i in prange(n):
                sumdata[i], sumerr[i], area[i], flag[i] = _sum_ellipse(
                    x[i], y[i], aper[i], data, mask, maskthresh, noise, segmap, seg_id[i], gain)
    else:
        aper_ann = np.empty((n, 5), np.float64)
        aper_ann[:, 0] = a
        aper_ann[:, 1] = b
        aper_ann[:, 2] = theta
        aper_ann[:, 3] = bkgann[0]
        aper_ann[:, 4] = bkgann[1]

        for i in prange(n):
            flux, fluxerr, area[i], flag[i] = _sum_ellipse(
                x[i], y[i], aper[i], data, mask, maskthresh, noise, segmap, seg_id[i], gain)
            if reject_outliers:
                bkgflux, bkgfluxerr, bkgarea, _ = _sum_ellipann_reject(
                    x[i], y[i], aper_ann[i], data, mask, maskthresh, noise, segmap, seg_id[i], gain, ignore_mask=True)
            else:
                bkgflux, bkgfluxerr, bkgarea, _ = _sum_ellipann(
                    x[i], y[i], aper_ann[i], data, mask, maskthresh, noise, segmap, seg_id[i], gain, ignore_mask=True)
            _area = area[i]
            if _area > 0 and bkgarea > 0:
                flux -= bkgflux/bkgarea*_area
                fluxerr = np.hypot(fluxerr, bkgfluxerr*_area/bkgarea)
            sumdata[i], sumerr[i] = flux, fluxerr

    return sumdata, sumerr, area, flag


@njitc(parallel=True)
def sum_ellipann(
        data: np.ndarray,
        x: float | np.ndarray,
        y: float | np.ndarray,
        a: float | np.ndarray,
        b: float | np.ndarray,
        theta: float | np.ndarray,
        rin: float | np.ndarray,
        rout: float | np.ndarray,
        var: float | np.ndarray = 0,
        err: float | np.ndarray = 0,
        gain: float = 0,
        mask: np.ndarray | None = None,
        maskthresh: float = 0,
        seg_id: int | np.ndarray | None = None,
        segmap: np.ndarray | None = None,
        reject_outliers: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parallel Numba port of :func:`sep.sum_ellipann`; always assumes subpix=0
    """
    x, y, noise, seg_id, sumdata, sumerr, area, flag = _prepare_arrays(data, x, y, var, err, mask, seg_id, segmap)
    n = x.size

    # Broadcast aperture parameters to x.shape
    aper = np.empty((n, 5), np.float64)
    aper[:, 0] = a
    aper[:, 1] = b
    aper[:, 2] = theta
    aper[:, 3] = rin
    aper[:, 4] = rout

    if reject_outliers:
        for i in prange(x.size):
            sumdata[i], sumerr[i], area[i], flag[i] = _sum_ellipann_reject(
                x[i], y[i], aper[i], data, mask, maskthresh, noise, segmap, seg_id[i], gain)
    else:
        for i in prange(x.size):
            sumdata[i], sumerr[i], area[i], flag[i] = _sum_ellipann(
                x[i], y[i], aper[i], data, mask, maskthresh, noise, segmap, seg_id[i], gain)

    return sumdata, sumerr, area, flag
