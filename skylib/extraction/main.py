"""
High-level source extraction interface.

:func:`~extract_sources()`: a wrapper around `~sep` background estimation and
source extraction functions.
"""

import numpy as np
from numpy.lib.recfunctions import append_fields
from scipy.ndimage import gaussian_filter
from astropy.stats import gaussian_fwhm_to_sigma, gaussian_sigma_to_fwhm
from astropy.convolution import Gaussian2DKernel, Kernel2D
from astropy.modeling.models import Gaussian2D
import sep

from ..calibration.background import estimate_background, sep_compatible
from .centroiding import centroid_iraf, centroid_iraf_masked


__all__ = [
    'extract_sources', 'histogram', 'auto_sat_level',
    'OBJ_MERGED', 'OBJ_TRUNC', 'OBJ_DOVERFLOW', 'OBJ_SINGU',
    'APER_TRUNC', 'APER_HASMASKED', 'APER_ALLMASKED', 'APER_NONPOSITIVE',
]


# Duplicate extraction flags
OBJ_MERGED = sep.OBJ_MERGED
OBJ_TRUNC = sep.OBJ_TRUNC
OBJ_DOVERFLOW = sep.OBJ_DOVERFLOW
OBJ_SINGU = sep.OBJ_SINGU
APER_TRUNC = sep.APER_TRUNC
APER_HASMASKED = sep.APER_HASMASKED
APER_ALLMASKED = sep.APER_ALLMASKED
APER_NONPOSITIVE = sep.APER_NONPOSITIVE


class AsymmetricGaussian2DKernel(Gaussian2DKernel):
    """
    Anisotropic Gaussian 2D filter kernel
    """
    def __init__(self, x_sigma: float, y_sigma: float, theta: float, **kwargs):
        """
        Create anisotropic 2D Gaussian kernel

        :param x_sigma: standard deviation along the major axis
        :param y_sigma: standard deviation along the minor axis
        :param theta: position angle of major axis with respect to the positive X axis, in degrees CCW
        :param kwargs: see `~astropy.convolution.Kernel2D`
        """
        self._model = Gaussian2D(1/(2*np.pi*x_sigma*y_sigma), 0, 0, x_sigma, y_sigma, theta*np.pi/180)
        i = int(np.ceil(8*max(x_sigma, y_sigma)))
        self._default_size = i if i % 2 else i + 1
        Kernel2D.__init__(self, **kwargs)
        self._truncation = abs(1 - self._array.sum())


def extract_sources(img: np.ndarray | np.ma.MaskedArray,
                    downsample: int = 2,
                    threshold: float = 2.5,
                    bkg_kw: dict[str, object] | None = None,
                    fwhm: float = 2.0,
                    ratio: float = 1,
                    theta: float = 0,
                    min_pixels: int = 5,
                    min_fwhm: float = 0.8,
                    max_fwhm: float = 50,
                    max_ellipticity: float = 2,
                    deblend: bool = True,
                    deblend_levels: int = 32,
                    deblend_contrast: float = 0.005,
                    clean: float | None = 1.0,
                    centroid: bool = True,
                    gain: float | None = None,
                    sat_img: np.ndarray | np.ma.MaskedArray | None = None,
                    discard_saturated: int = 0,
                    max_sources: int = 10000) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ma.MaskedArray, np.ma.MaskedArray, np.ma.MaskedArray]:
    """
    Extract sources from a calibrated (dark- and flat-corrected, etc.) image and return a table of their isophotal
    parameters

    This is a wrapper around :func:`sep.extract`, :func:`sep.winpos`, and
    :func:`skylib.calibration.background.estimate_background`.

    :param img: input 2D image array
    :param downsample: downsample input by the given factor to improve the reliability
    :param threshold: detection threshold in units of background RMS
    :param bkg_kw: optional keyword arguments to :func:`~skylib.calibration.background.estimate_background()`
    :param fwhm: estimated source FWHM in pixels; set to 0 to disable matched filter convolution
    :param ratio: minor to major Gaussian kernel axis ratio, 0 < ratio <= 1; ignored if `fwhm`=0; `ratio`=1 (default)
        means circular kernel; if `ratio`<1, it is assumed that `fwhm` corresponds to the minor axis of the kernel,
        which makes sense for sources elongated due to bad tracking
    :param theta: position angle of the Gaussian kernel major axis with respect to the positive X axis, in degrees CCW;
        ignored if `fwhm`=0 or `ratio`=1
    :param min_pixels: discard objects with less pixels above threshold
    :param min_fwhm: discard objects with smaller FWHM in pixels
    :param max_fwhm: discard objects with larger FWHM in pixels; 0 to disable
    :param max_ellipticity: discard objects with larger major to minor axis ratio; 0 to disable
    :param deblend: deblend overlapping sources
    :param deblend_levels: number of multi-thresholding levels to use; ignored if `deblend`=False
    :param deblend_contrast: fraction of the total flux to consider a component as a separate object; ignored if
        `deblend`=False
    :param clean: if not None and non-zero, perform cleaning with the given parameter
    :param centroid: obtain more accurate centroid positions after extraction using the windowed algorithm (SExtractor's
        XWIN_IMAGE, YWIN_IMAGE)
    :param gain: electrons to data units conversion factor; used to estimate photometric errors
    :param sat_img: optional image with non-zero values indicating saturated pixels; if provided, an extra column
        `saturated` is added that contains the number of saturated pixels in the source
    :param discard_saturated: if > 0 and `sat_img` is provided, discard sources with at least the given number of
        saturated pixels
    :param max_sources: maximum allowed number of detected sources

    :return:
        * record array containing isophotal parameters for each source; see :func:`~sep.extract` for a list of fields
        * background map array (ADUs), same shape as `img`
        * background RMS array (ADUs), same shape as `img`
    """
    threshold = float(threshold)
    if fwhm:
        fwhm = float(fwhm)
    else:
        fwhm = 0
    ratio = float(ratio)
    theta = float(theta)
    min_pixels = int(min_pixels)
    deblend = bool(int(deblend))
    deblend_levels = int(deblend_levels)
    deblend_contrast = float(deblend_contrast)
    if clean:
        clean = float(clean)
    else:
        clean = 0
    centroid = bool(int(centroid))
    if isinstance(gain, str):
        gain = float(gain)

    img = sep_compatible(img)

    # Estimate background
    if bkg_kw is None:
        bkg_kw = {}
    bkg, rms = estimate_background(img, **bkg_kw)

    # From now on, we'll work with a background-subtracted image
    det_img = img - bkg

    # Obtain filter kernel
    if fwhm:
        sigma = fwhm*gaussian_fwhm_to_sigma/downsample
        if ratio == 1:
            # Symmetric Gaussian kernel
            filter_kernel = Gaussian2DKernel(sigma)
        else:
            # Asymmetric Gaussian kernel
            filter_kernel = AsymmetricGaussian2DKernel(sigma/ratio, sigma, theta)
        filter_kernel.normalize()
        filter_kernel = filter_kernel.array
    else:
        # No filtering requested
        filter_kernel = None

    # Detect sources, obtain segmentation image to mark saturated sources
    # noinspection PyArgumentList
    if isinstance(det_img, np.ma.MaskedArray):
        _img = det_img.data
        _mask = det_img.mask
    else:
        _img = det_img
        _mask = None
    if isinstance(rms, np.ma.MaskedArray):
        if _mask is None:
            _mask = rms.mask
        else:
            _mask |= rms.mask
        _rms = rms.data
    else:
        _rms = rms

    # Downsample the image
    if downsample > 1:
        # Adjust extraction parameters
        min_pixels //= downsample**2

        # Before extraction, prefilter using Gaussian blur with sigma equal to downsampling factor
        _img = gaussian_filter(_img.astype(np.float32), sigma=downsample)

        # Rebin image and RMS map
        h, w = _img.shape
        width = w//downsample
        height = h//downsample
        if h/downsample % 1:
            _img = _img[:height*downsample]
            _rms = _rms[:height*downsample]
            if _mask is not None:
                _mask = _mask[:height*downsample]
            if sat_img is not None:
                sat_img = sat_img[:height*downsample]
        if w/downsample % 1:
            _img = _img[:, :width*downsample]
            _rms = _rms[:, :width*downsample]
            if _mask is not None:
                _mask = _mask[:, :width*downsample]
            if sat_img is not None:
                sat_img = sat_img[:, :width*downsample]
        _img = (_img.reshape((height, downsample, width, downsample)).sum(3).sum(1)/downsample**2).astype(_img.dtype)
        _rms = (_rms.reshape((height, downsample, width, downsample)).sum(3).sum(1)/downsample**2).astype(_rms.dtype)
        if _mask is not None:
            _mask = (_mask.reshape((height, downsample, width, downsample))
                     .sum(3).sum(1)/downsample**2).astype(_mask.dtype)
        if sat_img is not None:
            sat_img = (sat_img.reshape((height, downsample, width, downsample))
                       .sum(3).sum(1)/downsample**2).astype(sat_img.dtype)

    extract_kwargs = dict(
        err=_rms, mask=_mask, minarea=min_pixels, filter_kernel=filter_kernel, deblend_nthresh=deblend_levels,
        deblend_cont=deblend_contrast if deblend else 1.0, clean=bool(clean), clean_param=clean,
    )
    if sat_img is not None:
        sources, seg_img = sep.extract(_img, threshold, segmentation_map=True, **extract_kwargs)
    else:
        sources = sep.extract(_img, threshold, **extract_kwargs)
        seg_img = None
    del _img, _rms, _mask

    sources = append_fields(sources, 'saturated', np.zeros(len(sources), int), usemask=False)
    if seg_img is not None:
        # Count saturated pixels
        downsample2 = downsample**2
        for y, x in zip(*sat_img.nonzero()):
            i = seg_img[y//downsample, x//downsample]
            if i:
                sources[i - 1]['saturated'] += downsample2
        del seg_img

    # Exclude sources that couldn't be measured
    sources = sources[np.isfinite(sources['x']) & np.isfinite(sources['y']) &
                      np.isfinite(sources['a']) & np.isfinite(sources['b']) &
                      np.isfinite(sources['theta']) & np.isfinite(sources['flux'])]
    sources = sources[(sources['a'] > 0) & (sources['b'] > 0) & (sources['flux'] > 0)]

    if downsample > 1:
        # Rescale coordinates and sizes back
        sources['x'] *= downsample
        sources['y'] *= downsample
        sources['a'] *= downsample
        sources['b'] *= downsample

    # Discard saturated sources if requested
    if sat_img is not None and discard_saturated > 0:
        sources = sources[sources['saturated'] < discard_saturated]

    # Enforce hard threshold on the number of sources
    sources = sources[:max_sources]

    # Make sure that a >= b
    s = sources['a'] < sources['b']
    sources[s]['a'], sources[s]['b'] = sources[s]['b'], sources[s]['a']
    sources[s]['theta'] += np.pi/2
    sources['theta'] %= np.pi
    sources[sources['theta'] > np.pi/2]['theta'] -= np.pi

    # Discard sources with FWHM or ellipticity outside the limits
    if min_fwhm:
        sources = sources[sources['b'] >= min_fwhm*gaussian_fwhm_to_sigma]
    if max_fwhm:
        sources = sources[sources['a'] <= max_fwhm*gaussian_fwhm_to_sigma]
    if max_ellipticity:
        sources = sources[sources['a']/sources['b'] <= max_ellipticity]

    # Convert ADUs to electrons
    if gain:
        sources['cflux'] *= gain
        sources['flux'] *= gain

    # Convert to FITS origin convention
    sources['x'] += 1
    sources['y'] += 1

    if len(sources) and centroid:
        # Centroid sources using IRAF-like method
        radius = np.clip(0.5*gaussian_sigma_to_fwhm*sources['a'], 3, None)
        if isinstance(det_img, np.ma.MaskedArray) and det_img.mask is not False:
            centroid_iraf_masked(det_img.data, det_img.mask, sources['x'], sources['y'], radius)
        else:
            centroid_iraf(det_img, sources['x'], sources['y'], radius)

    return sources, bkg, rms


def histogram(data: np.ndarray | np.ma.MaskedArray, bins: str | int = 'auto') -> tuple[np.ndarray, float, float]:
    """
    Calculate image histogram with automatic data range and number of bins

    :param data: input array
    :param bins: either the number of histogram bins or algorithm to compute this number ("auto", "fd", "doane",
        "scott", "rice", "sturges", or "sqrt", see
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html, or "background"

    :return: 1D histogram array plus the left and right histogram boundaries
    """
    if isinstance(data, np.ma.MaskedArray):
        data = data.compressed()
    else:
        data = data.ravel()

    if data.size:
        min_bin = float(data.min(initial=None))
        max_bin = float(data.max(initial=None))

        if bins == 'background':
            from astropy.stats import SigmaClip
            from photutils.background import (
                Background2D, ModeEstimatorBackground, MADStdBackgroundRMS)
            mesh_size = np.floor(np.sqrt(data.size)/20)
            bkg = Background2D(
                data, (mesh_size, mesh_size), filter_size=(3, 3),
                sigma_clip=SigmaClip(sigma=3),
                bkg_estimator=ModeEstimatorBackground(),
                bkgrms_estimator=MADStdBackgroundRMS())
            # bin_width = bkg.background_rms_median*4/data.size**(1/3)

            bin_width = (bkg.background_rms_median*4)/30

            bins = np.floor((max_bin - min_bin) / bin_width)

        if isinstance(bins, int) and not (data % 1).any():
            # Integer data; don't use more bins than there are values
            bins = min(bins, int(max_bin - min_bin) + 1)

        if max_bin == min_bin:
            # Constant data, use unit bin size if the number of bins
            # is fixed or unit range otherwise
            if isinstance(bins, int):
                max_bin = min_bin + bins
            else:
                max_bin = min_bin + 1

        data = np.histogram(data, bins, (min_bin, max_bin))[0]
    else:
        # Empty or fully masked image
        data = np.array([], np.float32)
        min_bin, max_bin = 0.0, 65535.0

    return data, min_bin, max_bin


def auto_sat_level(data: np.ndarray | np.ma.MaskedArray) -> float | None:
    """
    Estimate saturation level based on the image histogram

    :param data: input array

    :return: empirical saturation level or None if no saturated pixels
        were found or not enough info to estimate the saturation level
    """
    # Compute a low-resolution image histogram
    n = 16
    coarse_hist, mn, mx = histogram(data, bins=n)
    if len(coarse_hist) < n:
        # Not enough data
        return
    n = len(coarse_hist)

    # Assume that the rightmost bin contains saturated pixels
    if coarse_hist[n - 1] <= coarse_hist[n - 2]:
        # Treat as no saturated pixels if the previous bin contains more pixels
        return

    # Find a more accurate saturation level value within the rightmost bin; this will be the right boundary of
    # the rightmost minimum to the left of the maximum of a 16x higher resolution histogram
    mn = mx - (mx - mn)/n
    n = 16
    hist = np.histogram(data, n, (mn, mx))[0]
    imax = np.argmax(hist)
    if imax:
        imin = imax - np.argmin(hist[:imax][::-1])
    else:
        imin = 0
    sat_level = mn + imin*(mx - mn)/n
    return sat_level
