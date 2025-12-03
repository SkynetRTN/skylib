"""
Getting standard info from FITS headers

:func:`~get_fits_time(): get exposure start/center/stop time from FITS header
"""

from datetime import datetime, timedelta
from typing import Optional, Tuple

from numpy import hypot
from astropy.io.fits import Header
from astropy.wcs import WCS


__all__ = [
    'str_to_datetime', 'get_fits_time', 'get_fits_exp_length', 'get_fits_gain',
]


def str_to_datetime(s: str) -> datetime | None:
    """
    Parse ISO datetime string and return a datetime object

    :param s: date/time string

    :return: datetime object, None if not a valid date/time string
    """
    t = None
    for fmt in (
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d',
            '%d/%m/%y',  # occurs in some old radio FITS files from SkyView
    ):
        try:
            t = datetime.strptime(s, fmt)
        except ValueError:
            pass
        else:
            break
    return t


def get_fits_time(hdr: Header, exp_length: Optional[float] = None) \
        -> Tuple[Optional[datetime], Optional[datetime], Optional[datetime]]:
    """
    Get exposure start, center, and stop times from FITS header

    :param hdr: FITS header
    :param exp_length: optional exposure length in seconds, e.g. from
        the database

    :return: exposure start, center, and stop times as datetime instances, None
        if unknown
    """
    t_start = t_cen = t_stop = None

    try:
        date_obs = hdr['DATE-OBS']
    except (KeyError, ValueError):
        date_obs = None
    else:
        # Normalize to string for parsing
        if date_obs is not None:
            date_obs = str(date_obs)
        time_obs = hdr.get('TIME-OBS')

        # Only append TIME-OBS if present and non-empty
        if date_obs is not None and 'T' not in date_obs and isinstance(time_obs, str) and time_obs:
            date_obs = date_obs + 'T' + time_obs

    t_start = str_to_datetime(date_obs) if date_obs is not None else None

    try:
        t_cen = str_to_datetime(hdr['DATE-CEN'])
    except (KeyError, ValueError):
        pass

    try:
        t_stop = str_to_datetime(hdr['DATE-END'])
    except (KeyError, ValueError):
        pass

    if None in (t_start, t_cen, t_stop):
        try:
            texp = float(hdr['EXPTIME'])
        except (KeyError, ValueError):
            try:
                texp = float(hdr['EXPOSURE'])
            except (KeyError, ValueError):
                texp = exp_length
        if texp is None:
            # Assume zero exposure length by default
            texp = 0

        if t_start is None:
            if t_cen is not None:
                t_start = t_cen - timedelta(seconds=texp/2)
            elif t_stop is not None:
                t_start = t_stop - timedelta(seconds=texp)

        if t_cen is None:
            if t_start is not None:
                t_cen = t_start + timedelta(seconds=texp/2)
            elif t_stop is not None:
                t_cen = t_stop - timedelta(seconds=texp/2)

        if t_stop is None:
            if t_cen is not None:
                t_stop = t_cen + timedelta(seconds=texp/2)
            elif t_start is not None:
                t_stop = t_start + timedelta(seconds=texp)

    return t_start, t_cen, t_stop


def get_fits_exp_length(hdr: Header) -> Optional[float]:
    """
    Get exposure length from FITS header

    :param hdr: FITS file header

    :return: exposure length in seconds; None if unknown
    """
    texp = None
    for name in ('EXPTIME', 'EXPOSURE'):
        try:
            texp = float(hdr[name])
        except (KeyError, ValueError):
            continue
        else:
            break
    return texp


def get_fits_gain(hdr: Header) -> float:
    """
    Get effective gain from FITS header

    :param hdr: FITS file header

    :return: effective gain in e-/ADU; None if unknown
    """
    gain = None
    for name in ('GAIN', 'EGAIN', 'EPERADU'):
        try:
            gain = float(hdr[name])
        except (KeyError, ValueError):
            continue
        else:
            break
    return gain


def get_fits_fov(hdr: Header) \
        -> Tuple[Optional[float], Optional[float], Optional[float],
                 Optional[float], int, int]:
    """
    Get FOV RA/Dec and radius from FITS header

    :param hdr: FITS file header

    :return: FOV center RA (hours), Dec (degrees), and radius (degrees),
        pixel scale (arcsec/pixel), and width and height (pixels); None
        if unknown
    """
    width, height = hdr.get('NAXIS1'), hdr.get('NAXIS2')
    ra0 = dec0 = radius = None
    # noinspection PyBroadException
    try:
        hdr['CRVAL1'] %= 360  # Ensure RA is in [0, 360) range
        wcs = WCS(hdr, relax=True)
        if not wcs.has_celestial:
            raise ValueError()
    except Exception:
        # No valid WCS in the header; try using MaxIm fields
        for name in ('OBJRA', 'TELRA', 'RA'):
            try:
                h, m, s = hdr[name].split(':')
                ra0 = int(h) + int(m)/60 + float(s.replace(',', '.'))/3600
            except (KeyError, ValueError):
                pass
            else:
                break
        for name in ('OBJDEC', 'TELDEC', 'DEC'):
            try:
                d, m, s = hdr[name].split(':')
                dec0 = (abs(int(d)) + int(m)/60 +
                        float(s.replace(',', '.'))/3600) * \
                       (1 - d.strip().startswith('-'))
            except (KeyError, ValueError):
                pass
            else:
                break
        scale = hdr.get('SECPIX')
    else:
        if width and height:
            ra0, dec0 = wcs.all_pix2world((width - 1)/2, (height - 1)/2, 0)
        else:
            ra0, dec0 = wcs.wcs.crval
        ra0 %= 360
        ra0 /= 15
        scales = wcs.proj_plane_pixel_scales()
        scale = (scales[0].to('arcsec').value + scales[1].to('arcsec').value)/2

    if scale is not None and width and height:
        radius = hypot(width, height)/2*scale/3600

    return ra0, dec0, radius, scale, width, height
