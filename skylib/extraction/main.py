"""
High-level source extraction interface.

:func:`~extract_sources()`: a wrapper around `~sep` background estimation and
source extraction functions.
"""

from __future__ import absolute_import, division, print_function

from typing import Any, Dict, Optional, Tuple, Union

from numpy import ceil, isfinite, ndarray, pi, zeros
from numpy.ma import MaskedArray
from numpy.lib.recfunctions import append_fields
from astropy.stats import gaussian_fwhm_to_sigma, gaussian_sigma_to_fwhm
from astropy.convolution import Gaussian2DKernel, Kernel2D
from astropy.modeling.models import Gaussian2D
import sep

from skylib.extraction.centroiding import centroid_sources

from ..calibration.background import estimate_background, sep_compatible


__all__ = [
    'extract_sources',
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
                    sat_img: Optional[Union[ndarray, MaskedArray]] = None) \
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
        to the positive X axis, in degrees CCW; ignored if `fwhm`=0 or `ratio`=1
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

    # Make sure that a >= b
    s = sources['a'] < sources['b']
    sources[s]['a'], sources[s]['b'] = sources[s]['b'], sources[s]['a']

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
