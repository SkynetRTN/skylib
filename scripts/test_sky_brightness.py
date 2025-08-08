#!/usr/bin/env python3
"""Sky brightness model testing script"""

import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body
from astropy.time import Time
import astropy.units as u
import matplotlib.pyplot as plt

from skylib.photometry.exposure import sky_brightness


def main():
    # Example location: Mauna Kea Observatory
    site = EarthLocation(lat=19.8207*u.deg, lon=-155.4681*u.deg, height=4205*u.m)

    # Example time: current time
    time = Time("2025-08-08 08:50:00")

    # Example coordinates: Zenith (straight up)
    coord = SkyCoord(alt=2*u.deg, az=0*u.deg, frame='altaz', obstime=time, location=site)

    sun = get_body("sun", time, site)
    sun_altaz = sun.transform_to(AltAz(obstime=time, location=site))
    moon = get_body("moon", time, site)
    moon_altaz = moon.transform_to(AltAz(obstime=time, location=site))
    print(f"Sun altitude: {sun_altaz.alt.deg:.2f}, azimuth: {sun_altaz.az.deg:.2f}")
    print(f"Moon altitude: {moon_altaz.alt.deg:.2f}, azimuth: {moon_altaz.az.deg:.2f}, "
          f"phase: {(1 - np.cos(sun.separation(moon).rad))/2:.2f}")

    # Calculate sky brightness
    brightness = sky_brightness(time, site, coord)

    print(f"Sky brightness at {coord.alt.deg:.1f} altitude and {coord.az.deg:.1f} azimuth")
    print(f"at {time.iso} from Mauna Kea Observatory is approximately {brightness:.2f} mag/arcsec²")

    def f(alt, az):
        return sky_brightness(
            time, site, SkyCoord(alt=alt*u.deg, az=az*u.deg, frame='altaz', obstime=time, location=site)
        )

    r = np.linspace(0, 90, 15)
    theta = np.linspace(0, 360, 36)
    r_grid, theta_grid = np.meshgrid(r, theta)

    z = np.empty_like(r_grid)
    for i,j in np.ndindex(r_grid.shape):
        z[i,j] = f(r_grid[i,j], theta_grid[i,j])

    plt.figure(figsize=(6, 6))
    img = plt.contourf(theta_grid, r_grid, z, 50, cmap='viridis_r')
    cbar = plt.colorbar(img)
    cbar.set_label("Sky Brightness (mag/arcsec²)")
    cbar.ax.invert_yaxis()
    plt.xlabel('Azimuth [°]')
    plt.ylabel('Elevation [°]')
    plt.title(f"Sun altitude: {sun_altaz.alt.deg:.2f}, azimuth: {sun_altaz.az.deg:.2f}\n"
              f"Moon altitude: {moon_altaz.alt.deg:.2f}, azimuth: {moon_altaz.az.deg:.2f}, "
              f"phase: {(1 - np.cos(sun.separation(moon).rad))/2:.2f}")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    # r = np.r_[0:10:0.01]
    # z = np.empty_like(r)
    # for i in range(len(r)):
    #     z[i] = f(r[i], 150)
    # plt.figure(figsize=(6, 6))
    # plt.plot(r, z, 'b-')
    # plt.show()


if __name__ == '__main__':
    main()
