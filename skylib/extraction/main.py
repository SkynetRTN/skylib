"""
High-level source extraction interface.

:func:`~extract_sources()`: a wrapper around `~sep` background estimation and
source extraction functions.
"""

from __future__ import absolute_import, division, print_function

from numpy import ceil, pi, zeros
from numpy.lib.recfunctions import append_fields
from astropy.stats import gaussian_fwhm_to_sigma
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
    def __init__(self, x_sigma, y_sigma, theta, **kwargs):
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


def extract_sources(img, threshold=2.5, bkg_kw=None, fwhm=2.0, ratio=1, theta=0,
                    min_pixels=5, deblend=True, deblend_levels=32,
                    deblend_contrast=0.005, clean=1.0, centroid=True, gain=None,
                    sat_img=None):
    """
    Extract sources from a calibrated (dark- and flat-corrected, etc.) image
    and return a table of their isophotal parameters

    This is a wrapper around :func:`sep.extract`, :func:`sep.winpos`, and
    :func:`skylib.calibration.background.estimate_background`.

    :param array_like img: input 2D image array
    :param float threshold: detection threshold in units of background RMS
    :param dict bkg_kw: optional keyword arguments to
        `~skylib.calibration.background.estimate_background()`
    :param float fwhm: estimated source FWHM in pixels; set to 0 to disable
        matched filter convolution
    :param float ratio: minor to major Gaussian kernel axis ratio, 0 < ratio <=
        1; ignored if `fwhm`=0; `ratio`=1 (default) means circular kernel;
        if `ratio`<1, it is assumed that `fwhm` corresponds to the minor axis
        of the kernel, which makes sense for sources elongated due to bad
        tracking
    :param float theta: position angle of the Gaussian kernel major axis with
        respect to the positive X axis, in degrees CCW; ignored if `fwhm`=0
        or `ratio`=1
    :param int min_pixels: discard objects with less pixels above threshold
    :param bool deblend: deblend overlapping sources
    :param int deblend_levels: number of multi-thresholding levels to use;
        ignored if `deblend`=False
    :param float deblend_contrast: fraction of the total flux to consider a
        component as a separate object; ignored if `deblend`=False
    :param float | None clean: if set and non-zero, perform cleaning with the
        given parameter
    :param bool centroid: obtain more accurate centroid positions after
        extraction using the windowed algorithm (SExtractor's XWIN_IMAGE,
        YWIN_IMAGE)
    :param float gain: electrons to data units conversion factor; used to
        estimate photometric errors
    :param array_like sat_img: optional image with non-zero values indicating
        saturated pixels; if provided, an extra column `saturated` is added that
        contains the number of saturated pixels in the source

    :return:
        record array containing isophotal parameters for each source; see
            :func:`~sep.extract` for a list of fields
        background map array, same shape as `img`
        background RMS array, same shape as `img`
    :rtype: tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray)
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
    sources, seg_img = sep.extract(
        det_img, threshold, err=rms, minarea=min_pixels, gain=gain,
        filter_kernel=filter_kernel, deblend_nthresh=deblend_levels,
        deblend_cont=deblend_contrast if deblend else 1.0,
        clean=bool(clean), clean_param=clean, segmentation_map=True)

    # Convert to FITS origin convention
    sources['x'] += 1
    sources['y'] += 1

    if len(sources) and centroid:
        # Centroid sources using the IRAF-like method
        sources['x'], sources['y'] = centroid_sources(
            det_img, sources['x'], sources['y'], sources['a'])

    sources = append_fields(
        sources, 'saturated', zeros(len(sources), int), usemask=False)
    if sat_img is not None:
        # Count saturated pixels
        for y, x in zip(*(seg_img.astype(bool) & sat_img).nonzero()):
            sources[seg_img[y, x] - 1]['saturated'] += 1

    return sources, bkg, rms
