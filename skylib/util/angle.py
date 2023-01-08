"""
Helper functions for working with angular quantities
"""

from typing import Iterable, Optional, Tuple, Union

import numpy as np


__all__ = ['angdist', 'average_radec']


def angdist(ra1_hours: float, dec1_degs: float, ra2_hours: float,
            dec2_degs: float) -> float:
    """
    Angular distance between two points on celestial sphere

    :param ra1_hours: right ascension of first point (hours)
    :param dec1_degs: declination of first point (degrees)
    :param ra2_hours: right ascension of second point (hours)
    :param dec2_degs: declination of second point (degrees)

    :return: angular distance in degrees
    """
    ra1 = np.deg2rad(ra1_hours)*15
    dec1 = np.deg2rad(dec1_degs)
    ra2 = np.deg2rad(ra2_hours)*15
    dec2 = np.deg2rad(dec2_degs)
    return 2*np.rad2deg(np.arcsin(np.sqrt(np.clip(
        np.sin((dec1 - dec2)/2)**2 +
        np.sin((ra1 - ra2)/2)**2*np.cos(dec1)*np.cos(dec2), 0, 1))))


def average_radec(ra_hours_or_radec: Iterable[Union[float,
                                                    Tuple[float, float]]],
                  dec_degs: Optional[Iterable[float]] = None) \
        -> Tuple[float, float]:
    """
    Mean right ascension and declination of multiple points on celestial sphere

    :param ra_hours_or_radec: array of right ascensions in hours
        (average_radec(ra, dec)) or array of RA/Dec pairs
        (average_radec(radec)) in hours and degrees, respectively
    :param dec_degs: array of declinations; unused in the second form
        (average_radec(radec))

    :return: average RA (hours) and Dec (degrees)
    """
    if dec_degs is None:
        ra_hours, dec_degs = np.transpose(ra_hours_or_radec)
        ra_degs = ra_hours*15
    else:
        ra_degs = np.asarray(ra_hours_or_radec)*15
    x = (np.cos(ra_degs)*np.cos(dec_degs)).mean()
    y = (np.sin(ra_degs)*np.cos(dec_degs)).mean()
    z = np.sin(dec_degs).mean()
    return (
        np.rad2deg(np.arctan2(y, x))/15 % 24,
        np.rad2deg(np.arctan2(z, np.hypot(x, y)))
    )
