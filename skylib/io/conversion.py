"""
Convert 2D image arrays to general image formats like JPEG and PNG.

:func:`~get_image()`: return a byte array containing image in the given format,
ready for saving to a file or sending over the Internet.
"""

from __future__ import absolute_import, division, print_function

from io import BytesIO

from numpy import asanyarray, clip, percentile, uint8
from astropy import visualization as apy_vis


__all__ = ['get_image']


def get_image(data, fmt='JPEG', norm='percentile', lo=None, hi=None,
              zcontrast=0.25, stretch='linear', a=None, bias=0.5, contrast=1,
              nsamples=1000, krej=2.5, max_iterations=5, cmap=None, dpi=100,
              **kwargs):
    """
    Return a byte array containing image in the given format

    Image scaling is done using `~astropy.visualization`. It includes
    normalization of the input data (mapping to [0, 1]) and stretching -
    optional non-linear mapping [0, 1] -> [0, 1] for contrast enhancement.
    A colormap can be applied to the normalized data. Conversion to the target
    image format is done by matplotlib or Pillow.

    :param array_like data: input 2D image data
    :param str fmt: output image format::
    :param str norm: data normalization mode::
        "manual": lower and higher clipping limits are set explicitly;
        "minmax": limits are set to the minimum and maximum data values;
        "percentile" (default): limits are set based on the specified fraction
            of pixels;
        "zscale": use IRAF ZScale algorithm
    :param int | float lo::
        for ``norm`` == "manual", lower data limit;
        for ``norm`` == "percentile", lower percentile clipping value,
            defaulting to 10;
        for ``norm`` == "zscale", lower limit on the number of rejected pixels,
            defaulting to 5
    :param int | float hi::
        for ``norm`` == "manual", upper data limit;
        for ``norm`` == "percentile", upper percentile clipping value,
            defaulting to 98;
        for ``norm`` == "zscale", upper limit on the number of rejected pixels,
            defaulting to data.size/2
    :param float zcontrast: for ``norm`` == "zscale", the scaling factor,
        0 < zcontrast < 1, defaulting to 0.25
    :param str stretch: [0, 1] -> [0, 1] mapping mode::
        "asinh": hyperbolic arcsine stretch y = asinh(x/a)/asinh(1/a)
        "contrast": linear bias/contrast-based stretch
            y = (x - bias)*contrast + 0.5;
        "exp": exponential stretch y = (a^x - 1)/(a - 1);
        "histeq": histogram equalization stretch;
        "linear" (default): direct mapping;
        "log": logarithmic stretch y = log(ax + 1)/log(a + 1);
        "power": power stretch y = x^a;
        "sinh": hyperbolic sine stretch y = sinh(x/a)/sinh(1/a);
        "sqrt": square root stretch y = \/x;
        "square": power stretch y = x^2
    :param float a: non-linear stretch parameter::
        for ``stretch`` == "asinh", the point of transition from linear to
            logarithmic behavior, 0 < a <= 1, defaulting to 0.1;
        for ``stretch`` == "exp", base of the exponent, a != 1, defaulting to
            1000;
        for ``stretch`` == "log", base of the logarithm minus 1, a > 0,
            defaulting to 1000;
        for ``stretch`` == "power", the power index, defaulting to 3;
        for ``stretch`` == "sinh", a > 0, defaulting to 1/3
    :param float bias: for ``stretch`` == "contrast", the bias parameter,
        defaulting to 0.5
    :param float contrast: for ``stretch`` == "contrast", the contrast
        parameter, defaulting to 1
    :param int nsamples: for ``stretch`` == "zscale", the number of points in
        the input array for determining scale factors, defaulting to 1000
    :param float krej: for ``stretch`` == "zscale", the sigma clipping factor,
        defaulting to 2.5
    :param int max_iterations: for ``stretch`` == "zscale", the maximum number
        of rejection iterations, defaulting to 5
    :param str cmap: optional matplotlib colormap name, defaulting to grayscale;
        when a non-grayscale colormap is specified, the conversion is always
        done by matplotlib, regardless of the availability of Pillow; see
        http://matplotlib.org/users/colormaps.html for more info on matplotlib
        colormaps and
            [name for name in matplotlib.cd.cmap_d.keys()
             if not name.endswith('_r')]
        to list the available colormap names
    :param int dpi: target image resolution in dots per inch
    :param kwargs: optional format-specific keyword arguments passed to Pillow,
        e.g. "quality" for JPEG; see
        `http://pillow.readthedocs.io/en/3.4.x/handbook/
        image-file-formats.html`_

    :return: a bytes object containing the image in the given format
    :rtype: bytes
    """
    data = asanyarray(data)

    # Normalize image data
    if norm == 'manual':
        if lo is None:
            raise ValueError(
                'Missing lower clipping boundary for norm="manual"')
        if hi is None:
            raise ValueError(
                'Missing upper clipping boundary for norm="manual"')
    elif norm == 'minmax':
        lo, hi = data.min(), data.max()
    elif norm == 'percentile':
        if lo is None:
            lo = 10
        elif not 0 <= lo <= 100:
            raise ValueError(
                'Lower clipping percentile must be in the [0,100] range')
        if hi is None:
            hi = 98
        elif not 0 <= hi <= 100:
            raise ValueError(
                'Upper clipping percentile must be in the [0,100] range')
        if hi < lo:
            raise ValueError(
                'Upper clipping percentile must be greater or equal to '
                'lower percentile')
        # noinspection PyTypeChecker
        lo, hi = percentile(data, [lo, hi])
    elif norm == 'zscale':
        if lo is None:
            lo = 5
        if hi is None:
            hi = 0.5
        else:
            hi /= data.size
        lo, hi = apy_vis.ZScaleInterval(
            nsamples, zcontrast, hi, lo, krej, max_iterations).get_limits(data)
    else:
        raise ValueError('Unknown normalization mode "{}"'.format(norm))
    # noinspection PyTypeChecker
    data = clip((data - lo)/(hi - lo), 0, 1)

    # Stretch the data
    if stretch == 'asinh':
        if a is None:
            a = 0.1
        apy_vis.AsinhStretch(a)(data, out=data)
    elif stretch == 'contrast':
        if bias != 0.5 or contrast != 1:
            apy_vis.ContrastBiasStretch(contrast, bias)(data, out=data)
    elif stretch == 'exp':
        if a is None:
            a = 1000
        apy_vis.PowerDistStretch(a)(data, out=data)
    elif stretch == 'histeq':
        apy_vis.HistEqStretch(data)(data, out=data)
    elif stretch == 'linear':
        pass
    elif stretch == 'log':
        if a is None:
            a = 1000
        apy_vis.LogStretch(a)(data, out=data)
    elif stretch == 'power':
        if a is None:
            a = 3
        apy_vis.PowerStretch(a)(data, out=data)
    elif stretch == 'sinh':
        if a is None:
            a = 1/3
        apy_vis.SinhStretch(a)(data, out=data)
    elif stretch == 'sqrt':
        apy_vis.SqrtStretch()(data, out=data)
    elif stretch == 'square':
        apy_vis.SquaredStretch()(data, out=data)
    else:
        raise ValueError('Unknown stretch mode "{}"'.format(stretch))

    buf = BytesIO()
    try:
        # Choose the backend for making an image
        if cmap is None:
            cmap = 'gray'
        if cmap == 'gray':
            try:
                # noinspection PyPackageRequirements
                from PIL import Image as pil_image
            except ImportError:
                pil_image = None
        else:
            pil_image = None

        if pil_image is not None:
            # Use Pillow for grayscale output if available; flip the image to
            # match the bottom-to-top FITS convention and convert from [0,1] to
            # unsigned byte
            pil_image.frombuffer(
                'L', data.shape[::-1], (data[::-1]*255 + 0.5).astype(uint8),
                'raw', 'L', 0, 1).save(buf, fmt, dpi=(dpi,)*2, **kwargs)
        else:
            # Use matplotlib for non-grayscale colormaps or if PIL is not
            # available
            # noinspection PyPackageRequirements
            from matplotlib import image as mpl_image
            if fmt.lower() == 'png':
                # PNG images are saved upside down by matplotlib, regardless of
                # the origin parameter
                data = data[::-1]
            mpl_image.imsave(
                buf, data, cmap=cmap, format=fmt, origin='lower', dpi=dpi)

        return buf.getvalue()
    finally:
        buf.close()
