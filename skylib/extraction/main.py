"""
High-level source extraction interface.

:func:`~extract_sources()`: a wrapper around `~sep` background estimation and
source extraction functions.
"""

from __future__ import absolute_import, division, print_function

from typing import Any, Dict, Optional, Tuple, Union

from numpy import (
    argmax, argmin, array, ceil, float32, histogram as numpy_histogram,
    isfinite, ndarray, pi, zeros)
from numpy.ma import MaskedArray
from numpy.lib.recfunctions import append_fields
from astropy.stats import gaussian_fwhm_to_sigma, gaussian_sigma_to_fwhm
from astropy.convolution import Gaussian2DKernel, Kernel2D
from astropy.modeling.models import Gaussian2D
import sep

from skylib.extraction.centroiding import centroid_sources

from ..calibration.background import estimate_background, sep_compatible


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
        :param theta: position angle of major axis with respect to the positive
            X axis, in degrees CCW
        :param kwargs: see `~astropy.convolution.Kernel2D`
        """
        self._model = Gaussian2D(
            1/(2*pi*x_sigma*y_sigma), 0, 0, x_sigma, y_sigma, theta*pi/180)
        i = int(ceil(8*max(x_sigma, y_sigma)))
        self._default_size = i if i % 2 else i + 1
        Kernel2D.__init__(self, **kwargs)
        self._truncation = abs(1 - self._array.sum())


def extract_sources(img: Union[ndarray, MaskedArray], threshold: float = 2.5,
                    bkg_kw: Optional[Dict[str, Any]] = None, fwhm: float = 2.0,
                    ratio: float = 1, theta: float = 0, min_pixels: int = 5,
                    min_fwhm: float = 0.8, max_fwhm: float = 10,
                    max_ellipticity: float = 2, deblend: bool = True,
                    deblend_levels: int = 32, deblend_contrast: float = 0.005,
                    clean: Optional[float] = 1.0, centroid: bool = True,
                    gain: float = None,
                    sat_img: Optional[Union[ndarray, MaskedArray]] = None,
                    discard_saturated: int = 0) \
        -> Union[Tuple[ndarray, ndarray, ndarray],
                 Tuple[MaskedArray, MaskedArray, MaskedArray]]:
    """
    Extract sources from a calibrated (dark- and flat-corrected, etc.) image
    and return a table of their isophotal parameters

    This is a wrapper around :func:`sep.extract`, :func:`sep.winpos`, and
    :func:`skylib.calibration.background.estimate_background`.

    :param img: input 2D image array
    :param threshold: detection threshold in units of background RMS
    :param bkg_kw: optional keyword arguments to
        `~skylib.calibration.background.estimate_background()`
    :param fwhm: estimated source FWHM in pixels; set to 0 to disable matched
        filter convolution
    :param ratio: minor to major Gaussian kernel axis ratio, 0 < ratio <= 1;
        ignored if `fwhm`=0; `ratio`=1 (default) means circular kernel;
        if `ratio`<1, it is assumed that `fwhm` corresponds to the minor axis
        of the kernel, which makes sense for sources elongated due to bad
        tracking
    :param theta: position angle of the Gaussian kernel major axis with respect
        to the positive X axis, in degrees CCW; ignored if `fwhm`=0 or
        `ratio`=1
    :param min_pixels: discard objects with less pixels above threshold
    :param min_fwhm: discard objects with smaller FWHM in pixels
    :param max_fwhm: discard objects with larger FWHM in pixels; 0 to disable
    :param max_ellipticity: discard objects with larger major to minor axis
        ratio; 0 to disable
    :param deblend: deblend overlapping sources
    :param deblend_levels: number of multi-thresholding levels to use;
        ignored if `deblend`=False
    :param deblend_contrast: fraction of the total flux to consider a component
        as a separate object; ignored if `deblend`=False
    :param clean: if not None and non-zero, perform cleaning with the given
        parameter
    :param centroid: obtain more accurate centroid positions after extraction
        using the windowed algorithm (SExtractor's XWIN_IMAGE, YWIN_IMAGE)
    :param gain: electrons to data units conversion factor; used to estimate
        photometric errors
    :param sat_img: optional image with non-zero values indicating saturated
        pixels; if provided, an extra column `saturated` is added that contains
        the number of saturated pixels in the source
    :param discard_saturated: if > 0 and `sat_img` is provided, discard sources
        with at least the given number of saturated pixels

    :return:
        record array containing isophotal parameters for each source; see
            :func:`~sep.extract` for a list of fields
        background map array (ADUs), same shape as `img`
        background RMS array (ADUs), same shape as `img`
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
        sigma = fwhm*gaussian_fwhm_to_sigma
        if ratio == 1:
            # Symmetric Gaussian kernel
            filter_kernel = Gaussian2DKernel(sigma)
        else:
            # Asymmetric Gaussian kernel
            filter_kernel = AsymmetricGaussian2DKernel(
                sigma/ratio, sigma, theta)
        filter_kernel.normalize()
        filter_kernel = filter_kernel.array
    else:
        # No filtering requested
        filter_kernel = None

    # Detect sources, obtain segmentation image to mark saturated sources
    # noinspection PyArgumentList
    if isinstance(det_img, MaskedArray):
        _img = det_img.data
        _mask = det_img.mask
    else:
        _img = det_img
        _mask = None
    if isinstance(rms, MaskedArray):
        if _mask is None:
            _mask = rms.mask
        else:
            _mask |= rms.mask
        _rms = rms.data
    else:
        _rms = rms
    sources, seg_img = sep.extract(
        _img, threshold, err=_rms, mask=_mask, minarea=min_pixels,
        filter_kernel=filter_kernel, deblend_nthresh=deblend_levels,
        deblend_cont=deblend_contrast if deblend else 1.0,
        clean=bool(clean), clean_param=clean, segmentation_map=True)

    sources = append_fields(
        sources, 'saturated', zeros(len(sources), int), usemask=False)
    if sat_img is not None:
        # Count saturated pixels
        for y, x in zip(*(seg_img.astype(bool) & sat_img).nonzero()):
            sources[seg_img[y, x] - 1]['saturated'] += 1

    # Exclude sources that couldn't be measured
    sources = sources[isfinite(sources['x']) & isfinite(sources['y']) &
                      isfinite(sources['a']) & isfinite(sources['b']) &
                      isfinite(sources['theta']) & isfinite(sources['flux'])]
    sources = sources[(sources['a'] > 0) & (sources['b'] > 0) &
                      (sources['flux'] > 0)]

    # Discard saturated sources if requested
    if sat_img is not None and discard_saturated > 0:
        sources = sources[sources['saturated'] < discard_saturated]

    # Make sure that a >= b
    s = sources['a'] < sources['b']
    sources[s]['a'], sources[s]['b'] = sources[s]['b'], sources[s]['a']
    sources[s]['theta'] += pi/2
    sources['theta'] %= pi
    sources[sources['theta'] > pi/2]['theta'] -= pi

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
        # Centroid sources using the IRAF-like method
        sources['x'], sources['y'] = centroid_sources(
            det_img, sources['x'], sources['y'],
            2*gaussian_sigma_to_fwhm*sources['a'])

    return sources, bkg, rms


def histogram(data: Union[ndarray, MaskedArray],
              bins: Union[str, int] = 'auto') -> Tuple[ndarray, float, float]:
    """
    Calculate image histogram with automatic data range and number of bins

    :param data: input array
    :param bins: either the number of histogram bins or algorithm to compute
        this number ("auto", "fd", "doane", "scott", "rice", "sturges", or
        "sqrt", see https://docs.scipy.org/doc/numpy/reference/generated/
        numpy.histogram.html

    :return: 1D histogram array plus the left and right histogram boundaries
    """
    if isinstance(data, MaskedArray):
        data = data.compressed()

    if data.size:
        min_bin = float(data.min(initial=None))
        max_bin = float(data.max(initial=None))
        if isinstance(bins, int) and not (data % 1).any():
            if max_bin - min_bin < 0x100:
                # 8-bit integer data; use 256 bins maximum
                bins = min(bins, 0x100)
            elif max_bin - min_bin < 0x10000:
                # 16-bit integer data; use 65536 bins maximum
                bins = min(bins, 0x10000)

        if max_bin == min_bin:
            # Constant data, use unit bin size if the number of bins
            # is fixed or unit range otherwise
            if isinstance(bins, int):
                max_bin = min_bin + bins
            else:
                max_bin = min_bin + 1

        data = numpy_histogram(data, bins, (min_bin, max_bin))[0]
    else:
        # Empty or fully masked image
        data = array([], float32)
        min_bin, max_bin = 0.0, 65535.0

    return data, min_bin, max_bin


def auto_sat_level(data: Union[ndarray, MaskedArray]) -> Optional[float]:
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
        print('Not enough data for auto sat level')
        return

    # Go 2/3 way to the right from the modal value
    binsize = (mx - mn)/n
    imax = argmax(coarse_hist)
    left = imax + 2*(n - imax)//3
    mn += left*binsize
    hist = coarse_hist[left:]
    if not len(hist):
        # Not enough data to find minimum
        print('No global minimum for auto sat level')
        return

    # Find the second mode in the right part of the histogram, which should
    # normally be the rightmost bin, and which corresponds to saturated pixels;
    # all variations are presumably introduced by applying dark and flat
    mn += argmax(hist)*binsize
    mx = mn + binsize

    # Find a more accurate saturation level value within this bin; this will be
    # the right boundary of the rightmost minimum to the left of the maximum
    # of a 8x higher resolution histogram ignoring data outside the modal bin
    n = 8
    hist = numpy_histogram(data[(data >= mn) & (data <= mx)], n, (mn, mx))[0]
    imax = argmax(hist)
    if imax:
        imin = imax - argmin(hist[:imax][::-1])
    else:
        imin = 0
    sat_level = mn + imin*(mx - mn)/n
    print('Auto sat level: {} [{}, {}]; coarse hist: {}, fine hist: {}'
          .format(sat_level, mn, mx, coarse_hist, hist))
    return sat_level
