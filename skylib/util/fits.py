"""
Getting standard info from FITS headers

:func:`~get_fits_time(): get exposure start/center/stop time from FITS header
"""

from __future__ import absolute_import, division, print_function

from datetime import datetime, timedelta


__all__ = ['get_fits_time', 'str_to_datetime']


def str_to_datetime(s):
    """
    Parse ISO datetime string and return a datetime object

    :param str s: date/time string

    :return: datetime object, None if not a valid date/time string
    :rtype: datetime.datetime | None
    """
    t = None
    for fmt in ('%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d'):
        try:
            t = datetime.strptime(s, fmt)
        except ValueError:
            pass
        else:
            break
    return t


def get_fits_time(hdr, exp_length=None):
    """
    Get exposure start, center, and stop times from FITS header

    :param :class:`astropy.io.fits.Header` hdr: FITS header
    :param float exp_length: optional exposure length in seconds, e.g. from
        the database

    :return: exposure start, center, and stop times as datetime instances, None
        if unknown
    :rtype: tuple(3)[:class:`datetime.datetime` | None]
    """
    t_start = t_cen = t_stop = None

    try:
        t_start = str_to_datetime(hdr['DATE-OBS'])
    except KeyError:
        pass

    try:
        t_cen = str_to_datetime(hdr['DATE-CEN'])
    except KeyError:
        pass

    try:
        t_stop = str_to_datetime(hdr['DATE-END'])
    except KeyError:
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
