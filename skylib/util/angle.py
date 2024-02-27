"""
Helper functions for working with angular quantities
"""

from typing import Iterable

import numpy as np
from numba import njit


__all__ = ['angdist', 'average_radec']


@njit(cache=True)
def angdist(ra1_hours: float | np.ndarray, dec1_degs: float | np.ndarray,
            ra2_hours: float | np.ndarray, dec2_degs: float | np.ndarray) -> float | np.ndarray:
    """
    Angular distance between two points on celestial sphere

    :param ra1_hours: right ascension of first point (hours)
    :param dec1_degs: declination of first point (degrees)
    :param ra2_hours: right ascension of second point (hours)
    :param dec2_degs: declination of second point (degrees)

    :return: angular distance in degrees
    """
    ra1 = ra1_hours*(np.pi/12)
    dec1 = np.deg2rad(dec1_degs)
    ra2 = ra2_hours*(np.pi/12)
    dec2 = np.deg2rad(dec2_degs)
    return 2*np.rad2deg(np.arcsin(np.sqrt(
        np.sin((dec1 - dec2)/2)**2 + np.sin((ra1 - ra2)/2)**2*np.cos(dec1)*np.cos(dec2))))


@njit(cache=True)
def average_radec(radec: np.ndarray) -> tuple[float, float]:
    """
    Mean right ascension and declination of multiple points on celestial sphere

    :param radec: (2xN) array of RA/Dec pairs (average_radec(radec)) in hours and degrees, respectively

    :return: average RA (hours) and Dec (degrees)
    """
    ra = radec[:, 0]*(np.pi/12)
    dec = np.deg2rad(radec[:, 1])
    cos_dec = np.cos(dec)
    x = (np.cos(ra)*cos_dec).mean()
    y = (np.sin(ra)*cos_dec).mean()
    z = np.sin(dec).mean()
    return np.arctan2(y, x)*(12/np.pi) % 24, np.rad2deg(np.arctan2(z, np.hypot(x, y)))
