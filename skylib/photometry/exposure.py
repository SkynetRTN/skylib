"""
Exposure time calculator based on the sky brightness and the desired signal-to-noise ratio.
"""

from datetime import datetime
import warnings

import numpy as np
from numba import njit
from scipy.optimize import fsolve, least_squares
from astropy.time import Time
from astropy.coordinates import AltAz, EarthLocation, NonRotationTransformationWarning, SkyCoord, get_body

from ..util.angle import airmass_for_el


__all__ = [
    "sky_brightness", "calibrate_sky_model",
    "exptime_for_mag_and_snr", "exptime_for_mag_and_counts", "mag_for_exptime_and_snr", "snr_for_mag_and_exptime",
    "flux15_for_exptime_mag_and_snr", "planck_law", "planck_law_normalized", "dust_extinction",
]


@njit(nogil=True, cache=True)
def hg_scattering(g: float, eps: float, eta: float, cos_theta: float) -> float:
    """
    Henyey--Greenstein scattering function

    :param g: Asymmetry parameter (0 = isotropic, 1 = pure forward scattering).
    :param eps: Backscatter damping parameter.
    :param eta: Proportion of Rayleigh to aerosol scattering.
    :param cos_theta: Cosine of the scattering angle.

    :return: Scattering function value.
    """
    return eps + (1 - eps)*((1 - eta)*(1 - g)**3/(1 + g**2 - 2*g*cos_theta)**1.5 + eta*(1 + cos_theta**2)/2)


@njit(nogil=True, cache=True)
def full_scattering(
        mag: float,
        scatter_asymmetry: float,
        backscatter_damping: float,
        rayleigh_to_aerosol_weight: float,
        extinction_coeff: float,
        multiple_scattering_exponent: float,
        cos_theta: float,
        airmass: float) -> float:
    """
    Apply the full Sun or Moon scattering model to a given magnitude.

    :param mag: Starting magnitude (zero point).
    :param scatter_asymmetry: Henyey--Greenstein aerosol scattering asymmetry parameter g.
    :param backscatter_damping: Backscatter damping parameter.
    :param rayleigh_to_aerosol_weight: Proportion of Rayleigh scattering compared to aerosol.
    :param extinction_coeff: Atmospheric extinction coefficient in magnitudes per airmass.
    :param multiple_scattering_exponent: Empirical multiple scattering exponent.
    :param cos_theta: Cosine of the angle between the light source (Sun or Moon) and the target.
    :param airmass: Airmass of the target.

    :return: Adjusted magnitude.
    """
    # Scattering anisotropy
    mag -= 2.5*np.log10(
        hg_scattering(scatter_asymmetry, backscatter_damping, rayleigh_to_aerosol_weight, cos_theta)
        #/ hg_scattering(scatter_asymmetry, backscatter_damping, rayleigh_to_aerosol_weight, 0)
    )

    # Airmass scaling and multiple scattering
    mag -= 2.5*multiple_scattering_exponent*np.log10(airmass)

    # Linear extinction
    mag += extinction_coeff*(min(airmass, 7) - 1)

    return mag


def sky_brightness(
        t: Time | datetime | str,
        site: EarthLocation,
        target: SkyCoord,
        twilight_sky_brightness: float = 17.56,
        twilight_sun_elevation: float = -9.54,
        twilight_coeff: float = -0.057,
        scatter_asymmetry: float = 0.8,
        backscatter_damping: float = 0.05,
        rayleigh_to_aerosol_weight: float = 0.5,
        multiple_scattering_exponent: float = 1.3,
        extinction_coeff: float = 0.15,
        moonlight_zero_point: float = 18.0,
) -> float:
    """
    Estimate the sky brightness based on the Sun's elevation, Moon's phase, and the mutual position of Sun, Moon, and
    target.

    :param t: Observation time.
    :param site: Observation site.
    :param target: Target coordinates.
    :param twilight_sky_brightness: Twilight sky brightness at zenith at `twilight_sun_elevation` for the given bandpass
        in magnitudes per arcsecond squared.
    :param twilight_sun_elevation: Reference sun elevation for `twilight_sky_brightness` in degrees.
    :param twilight_coeff: Second-order twilight sky brightness factor for elevations < `twilight_sun_elevation` (< 0).
    :param scatter_asymmetry: Aerosol scattering asymmetry parameter g in the Henyey--Greenstein model (> 0 and < 1).
    :param backscatter_damping: Backscatter damping parameter (small value >0).
    :param rayleigh_to_aerosol_weight: Proportion of Rayleigh scattering compared to aerosol (0 to 1, 0 = pure Raleigh,
        1 = pure Mie).
    :param multiple_scattering_exponent: Empirical multiple scattering exponent (> 0).
    :param extinction_coeff: Atmospheric extinction coefficient for the given bandpass in magnitudes per airmass (> 0).
    :param moonlight_zero_point: Zenith sky brightness for the given bandpass in magnitudes per arcsecond squared at
        full Moon at zenith.

    :returns: Estimated sky brightness in magnitudes per arcsecond squared.
    """
    if isinstance(t, (datetime, str)):
        t = Time(t)
    elif not isinstance(t, Time):
        raise TypeError("t must be a Time object or a datetime or a string")

    sun = get_body("sun", t, site)
    sun_altaz = sun.transform_to(AltAz(obstime=t, location=site))
    sun_h = max(sun_altaz.alt.deg, -18)

    # Dependence on target azimuth and elevation
    target_altaz = target.transform_to(AltAz(obstime=t, location=site))
    airmass = airmass_for_el(target_altaz.alt.deg)/airmass_for_el(90)

    # Nighttime and twilight sky at zenith
    dz = twilight_sun_elevation - sun_h
    sky_mag = twilight_sky_brightness + dz
    if dz > 0:
        sky_mag += twilight_coeff*dz**2
    cos_theta = (  # Sun-target separation
        np.sin(target_altaz.alt.rad)*np.sin(sun_altaz.alt.rad) +
        np.cos(target_altaz.alt.rad)*np.cos(sun_altaz.alt.rad)*np.cos((target_altaz.az - sun_altaz.az).rad)
    )
    sky_mag = full_scattering(
        sky_mag, scatter_asymmetry, backscatter_damping, rayleigh_to_aerosol_weight, extinction_coeff,
        multiple_scattering_exponent, cos_theta, airmass
    )

    # Moonlight contribution
    moon = get_body("moon", t, site)
    moon_altaz = moon.transform_to(AltAz(obstime=t, location=site))
    moon_h = moon_altaz.alt.deg
    if moon_h <= 0:
        # No moonlight contribution
        return sky_mag
    moon_phase = (1 - np.cos(sun.separation(moon).rad))/2
    if moon_phase <= 0:
        # We won't handle solar eclipses here
        return sky_mag
    cos_theta = (  # Moon-target separation
        np.sin(target_altaz.alt.rad)*np.sin(moon_altaz.alt.rad) +
        np.cos(target_altaz.alt.rad)*np.cos(moon_altaz.alt.rad)*np.cos((target_altaz.az - moon_altaz.az).rad)
    )
    moon_mag = full_scattering(
        moonlight_zero_point - 2.5*np.log10(moon_phase), scatter_asymmetry, backscatter_damping,
        rayleigh_to_aerosol_weight, extinction_coeff, multiple_scattering_exponent, cos_theta, airmass
    )
    moon_mag += extinction_coeff*airmass_for_el(moon_h)/airmass_for_el(90)  # Moonlight extinction

    return -2.5*np.log10(10**(-0.4*sky_mag) + 10**(-0.4*moon_mag))


# noinspection PyIncorrectDocstring
def calibrate_sky_model(
        t: Time, site: EarthLocation, targets: SkyCoord, mags: np.ndarray,
        twilight_sky_brightness: float = 17.56, fix_twilight_sky_brightness: bool = False,
        twilight_sun_elevation: float = -9.54, fix_twilight_sun_elevation: bool = False,
        twilight_coeff: float = -0.057, fix_twilight_coeff: bool = False,
        scatter_asymmetry: float = 0.8, fix_scatter_asymmetry: bool = False,
        backscatter_damping: float = 0.05, fix_backscatter_damping: bool = False,
        rayleigh_to_aerosol_weight: float = 0.5, fix_rayleigh_to_aerosol_weight: bool = False,
        multiple_scattering_exponent: float = 1.3, fix_multiple_scattering_exponent: bool = False,
        extinction_coeff: float = 0.15, fix_extinction_coeff: bool = False,
        moonlight_zero_point: float = 18.0, fix_moonlight_zero_point: bool = False,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    """
    Calculate the three sky background model parameters given a set of measured sky background magnitudes.

    To reliably estimate all model parameters, the input dataset must contain observations with considerably high
    background levels and obtained both with and without the effect of the Moon and both at twilight and full night.

    :param t: Array of observation epochs.
    :param site: Observation site.
    :param targets: Array of target coordinates, same shape as `t`.
    :param mags: Array of measured sky background magnitudes, same shape as `t`.

    Other params: see `sky_brightness()`.

    :returns: Estimated sky background model parameters
    """
    def func(x):
        # Parse parameters that can vary
        i = 0
        if fix_twilight_sky_brightness:
            tsb = twilight_sky_brightness
        else:
            tsb = x[i]
            i += 1
        if fix_twilight_sun_elevation:
            tse = twilight_sun_elevation
        else:
            tse = x[i]
            i += 1
        if fix_twilight_coeff:
            tc = twilight_coeff
        else:
            tc = x[i]
            i += 1
        if fix_scatter_asymmetry:
            sa = scatter_asymmetry
        else:
            sa = x[i]
            i += 1
        if fix_backscatter_damping:
            bd = backscatter_damping
        else:
            bd = x[i]
            i += 1
        if fix_rayleigh_to_aerosol_weight:
            rtaw = rayleigh_to_aerosol_weight
        else:
            rtaw = x[i]
            i += 1
        if fix_multiple_scattering_exponent:
            mse = multiple_scattering_exponent
        else:
            mse = x[i]
            i += 1
        if fix_extinction_coeff:
            ec = extinction_coeff
        else:
            ec = x[i]
            i += 1
        if fix_moonlight_zero_point:
            mzp = moonlight_zero_point
        else:
            mzp = x[i]
            i += 1

        res = mags.copy()
        for i, (ti, ci) in enumerate(zip(t, targets)):
            res[i] -= sky_brightness(
                ti, site, ci, twilight_sky_brightness=tsb, twilight_sun_elevation=tse, twilight_coeff=tc,
                scatter_asymmetry=sa, backscatter_damping=bd, rayleigh_to_aerosol_weight=rtaw,
                multiple_scattering_exponent=mse, extinction_coeff=ec, moonlight_zero_point=mzp,
            )
        return res

    warnings.filterwarnings('ignore', category=NonRotationTransformationWarning)

    x0, bounds_lo, bounds_hi = [], [], []
    if not fix_twilight_sky_brightness:
        x0.append(twilight_sky_brightness)
        bounds_lo.append(10)
        bounds_hi.append(25)
    if not fix_twilight_sun_elevation:
        x0.append(twilight_sun_elevation)
        bounds_lo.append(-18)
        bounds_hi.append(-6)
    if not fix_twilight_coeff:
        x0.append(twilight_coeff)
        bounds_lo.append(-1)
        bounds_hi.append(0)
    if not fix_scatter_asymmetry:
        x0.append(scatter_asymmetry)
        bounds_lo.append(0)
        bounds_hi.append(1)
    if not fix_backscatter_damping:
        x0.append(backscatter_damping)
        bounds_lo.append(0)
        bounds_hi.append(0.1)
    if not fix_rayleigh_to_aerosol_weight:
        x0.append(rayleigh_to_aerosol_weight)
        bounds_lo.append(0)
        bounds_hi.append(1)
    if not fix_multiple_scattering_exponent:
        x0.append(multiple_scattering_exponent)
        bounds_lo.append(1)
        bounds_hi.append(2)
    if not fix_extinction_coeff:
        x0.append(extinction_coeff)
        bounds_lo.append(0.12)
        bounds_hi.append(0.2)
    if not fix_moonlight_zero_point:
        x0.append(moonlight_zero_point)
        bounds_lo.append(17)
        bounds_hi.append(20)

    if x0:
        params = least_squares(func, x0, bounds=(bounds_lo, bounds_hi)).x
        p = 0
        if not fix_twilight_sky_brightness:
            twilight_sky_brightness = params[p]
            p += 1
        if not fix_twilight_sun_elevation:
            twilight_sun_elevation = params[p]
            p += 1
        if not fix_twilight_coeff:
            twilight_coeff = params[p]
            p += 1
        if not fix_scatter_asymmetry:
            scatter_asymmetry = params[p]
            p += 1
        if not fix_backscatter_damping:
            backscatter_damping = params[p]
            p += 1
        if not fix_rayleigh_to_aerosol_weight:
            rayleigh_to_aerosol_weight = params[p]
            p += 1
        if not fix_multiple_scattering_exponent:
            multiple_scattering_exponent = params[p]
            p += 1
        if not fix_extinction_coeff:
            extinction_coeff = params[p]
            p += 1
        if not fix_moonlight_zero_point:
            moonlight_zero_point = params[p]
            p += 1

    return (
        twilight_sky_brightness, twilight_sun_elevation, twilight_coeff, scatter_asymmetry, backscatter_damping,
        rayleigh_to_aerosol_weight, multiple_scattering_exponent, extinction_coeff, moonlight_zero_point,
    )


def exptime_for_mag_and_snr(
        point_source: bool,
        mag: float,
        snr: float,
        read_noise: float,
        sky: float,
        dark: float,
        flux15: float,
        pixsize: float,
        seeing: float = 2,
        aper: float | None = None,
) -> float:
    """
    Calculate the exposure time for a given star magnitude and desired signal-to-noise ratio.

    :param point_source: whether the object is a point or extended source.
    :param mag: object brightness in mag for point sources; surface brightness in mag/arcsec^2 for extended sources.
    :param snr: desired signal-to-noise ratio (per pixel for extended sources).
    :param read_noise: read noise in electrons per pixel.
    :param sky: sky brightness in magnitudes per arcsec^2.
    :param dark: dark current in electrons per pixel per second.
    :param flux15: flux for a 15th magnitude object in electrons per second: flux15 = 10**(-0.4*(15 - zp)).
    :param pixsize: pixel size in arcsec.
    :param seeing: seeing in arcsec. Unused for extended sources.
    :param aper: photometric aperture diameter in arcsec; 2*`seeing` if omitted. Unused for extended sources.

    :return: exposure time in seconds.
    """
    background = dark + flux15*pixsize**2*10**(-0.4*(sky - 15))  # total background flux per pixel per second

    if point_source:
        if aper is None:
            aper = 2*seeing
        rad = aper/2/pixsize
        counts = flux15*10**(-0.4*(mag - 15))*(1 - np.exp(-0.5*(rad/seeing*2.35*pixsize)**2))
        total_counts = counts + background*np.pi*rad**2
    else:
        counts = flux15*pixsize**2*10**(-0.4*(mag - 15))
        total_counts = counts + background

    a = counts**2
    b = -total_counts*snr**2
    c = -(read_noise*snr)**2
    return (-b + np.sqrt(b**2 - 4*a*c))/(2*a)


def exptime_for_mag_and_counts(
        point_source: bool,
        mag: float,
        target_counts: float,
        sky: float,
        dark: float,
        flux15: float,
        pixsize: float,
        seeing: float = 2,
        aper: float | None = None,
) -> float:
    """
    Calculate the exposure time for a given star magnitude and desired electron count.

    :param point_source: whether the object is a point or extended source.
    :param mag: object brightness in mag for point sources; surface brightness in mag/arcsec^2 for extended sources.
    :param target_counts: desired electrons per pixel (not including blank level).
    :param sky: sky brightness in magnitudes per arcsec^2.
    :param dark: dark current in electrons per pixel per second.
    :param flux15: flux for a 15th magnitude object in electrons per second: flux15 = 10**(-0.4*(15 - zp)).
    :param pixsize: pixel size in arcsec.
    :param seeing: seeing in arcsec. Unused for extended sources.
    :param aper: photometric aperture diameter in arcsec; 2*`seeing` if omitted. Unused for extended sources.

    :return: exposure time in seconds.
    """
    background = dark + flux15*pixsize**2*10**(-0.4*(sky - 15))  # total background flux per pixel per second

    if point_source:
        if aper is None:
            aper = 2*seeing
        rad = aper/2/pixsize
        counts = flux15*10**(-0.4*(mag - 15))*(1 - np.exp(-0.5*(rad/seeing*2.35*pixsize)**2))
        total_counts = counts + background*np.pi*rad**2
    else:
        counts = flux15*pixsize**2*10**(-0.4*(mag - 15))
        total_counts = counts + background

    return target_counts/total_counts


def mag_for_exptime_and_snr(
        point_source: bool,
        texp: float,
        snr: float,
        read_noise: float,
        sky: float,
        dark: float,
        flux15: float,
        pixsize: float,
        seeing: float = 2,
        aper: float | None = None,
) -> float:
    """
    Calculate the object magnitude for a given exposure time and desired signal-to-noise ratio.

    :param point_source: whether the object is a point or extended source.
    :param texp: exposure time in seconds.
    :param snr: desired signal-to-noise ratio (per pixel for extended sources).
    :param read_noise: read noise in electrons per pixel.
    :param sky: sky brightness in magnitudes per arcsec^2.
    :param dark: dark current in electrons per pixel per second.
    :param flux15: flux for a 15th magnitude object in electrons per second: flux15 = 10**(-0.4*(15 - zp)).
    :param pixsize: pixel size in arcsec.
    :param seeing: seeing in arcsec. Unused for extended sources.
    :param aper: photometric aperture diameter in arcsec; 2*`seeing` if omitted. Unused for extended sources.

    :return: object brightness in mag for point sources; surface brightness in mag/arcsec^2 for extended sources.
    """
    background = dark + flux15*pixsize**2*10**(-0.4*(sky - 15))  # total background flux per pixel per second

    if point_source:
        if aper is None:
            aper = 2*seeing
        rad = aper/2/pixsize
        sky_counts = background*np.pi*rad**2
        aperfunc = 1 - np.exp(-0.5*(rad/seeing*2.35)**2)
    else:
        sky_counts = background
        aperfunc = pixsize**2

    a = texp**2
    b = -texp*snr**2
    c = -(texp*sky_counts + read_noise**2)*snr**2
    counts = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
    return 15 - 2.5*np.log10(counts/flux15/aperfunc)


def snr_for_mag_and_exptime(
        point_source: bool,
        mag: float,
        texp: float,
        read_noise: float,
        sky: float,
        dark: float,
        flux15: float,
        pixsize: float,
        seeing: float = 2,
        aper: float | None = None,
) -> float:
    """
    Calculate the signal-to-noise ratio for a given magnitude and exposure time.

    :param point_source: whether the object is a point or extended source.
    :param mag: object brightness in mag for point sources; surface brightness in mag/arcsec^2 for extended sources.
    :param texp: exposure time in seconds.
    :param read_noise: read noise in electrons per pixel.
    :param sky: sky brightness in magnitudes per arcsec^2.
    :param dark: dark current in electrons per pixel per second.
    :param flux15: flux for a 15th magnitude object in electrons per second: flux15 = 10**(-0.4*(15 - zp)).
    :param pixsize: pixel size in arcsec.
    :param seeing: seeing in arcsec. Unused for extended sources.
    :param aper: photometric aperture diameter in arcsec; 2*`seeing` if omitted. Unused for extended sources.

    :return: signal-to-noise ratio (per pixel for extended sources).
    """
    background = dark + flux15*pixsize**2*10**(-0.4*(sky - 15))  # total background flux per pixel per second

    if point_source:
        if aper is None:
            aper = 2*seeing
        rad = aper/2/pixsize
        counts = flux15*10**(-0.4*(mag - 15))*(1 - np.exp(-0.5*(rad/seeing*2.35*pixsize)**2))
        total_counts = counts + background*np.pi*rad**2
    else:
        counts = flux15*pixsize**2*10**(-0.4*(mag - 15))
        total_counts = counts + background

    return counts*texp/np.sqrt(total_counts*texp + read_noise**2)


def flux15_for_exptime_mag_and_snr(
        point_source: bool,
        texp: float,
        mag: float,
        snr: float,
        read_noise: float,
        sky: float,
        dark: float,
        pixsize: float,
        seeing: float = 2,
        aper: float | None = None,
) -> float:
    """
    Calculate the flux for a 15th magnitude object for a given exposure time, object magnitude, and signal-to-noise
    ratio. Used to calibrate the photometric system of an instrument.

    :param point_source: whether the object is a point or extended source.
    :param texp: exposure time in seconds.
    :param mag: object brightness in mag for point sources; surface brightness in mag/arcsec^2 for extended sources.
    :param snr: desired signal-to-noise ratio (per pixel for extended sources).
    :param read_noise: read noise in electrons per pixel.
    :param sky: sky brightness in magnitudes per arcsec^2.
    :param dark: dark current in electrons per pixel per second.
    :param pixsize: pixel size in arcsec.
    :param seeing: seeing in arcsec. Unused for extended sources.
    :param aper: photometric aperture diameter in arcsec; 2*`seeing` if omitted. Unused for extended sources.

    :return: flux for a 15th magnitude object in electrons per second.
    """
    # First, assume no sky background as we don't know flux15 yet and thus cannot convert to electrons
    if point_source:
        if aper is None:
            aper = 2*seeing
        rad = aper/2/pixsize
        sky_counts = dark*np.pi*rad**2
        aperfunc = 1 - np.exp(-0.5*(rad/seeing*2.35)**2)
    else:
        sky_counts = dark
        aperfunc = pixsize**2

    a = texp**2
    b = -texp*snr**2
    c = -(texp*sky_counts + read_noise**2)*snr**2
    counts = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
    flux15 = counts/10**(-0.4*(mag - 15))/aperfunc  # initial guess

    # Solve for flux15
    return float(fsolve(
        lambda x:
            mag - mag_for_exptime_and_snr(point_source, texp, snr, read_noise, sky, dark, x, pixsize, seeing, aper),
        flux15)[0])


# Planck's law; used by the brightness model with blackbody spectral component
planck_law_h = 6.62607015e-34  # Planck's constant (SI)
planck_law_c = 2.99792458e8  # Speed of light  (SI)
planck_law_k = 1.380649e-23  # Boltzmann constant (SI)
planck_law_b = 2.897771955e-3  # Wien's displacement constant (SI)


def planck_law(wavelength: float, temperature: float) -> float:
    """
    Calculate the spectral radiance of a black body at a given wavelength and temperature using Planck's law.

    :param wavelength: Wavelength in meters.
    :param temperature: Temperature in Kelvin.

    :return: Spectral radiance in W/(m^2 * sr * m).
    """
    return (2*planck_law_h*planck_law_c**2)/(wavelength**5) \
        / (np.exp(planck_law_h*planck_law_c/(wavelength*planck_law_k*temperature)) - 1)


def planck_law_normalized(wavelength: float, temperature: float) -> float:
    """
    Calculate the normalized spectral radiance of a black body at a given wavelength and temperature using Planck's law.

    :param wavelength: Wavelength in meters.
    :param temperature: Temperature in Kelvin.

    :return: Normalized spectral radiance.
    """
    return planck_law(wavelength, temperature)/planck_law(planck_law_b/temperature, temperature)


def dust_extinction(filter_wavelength: float, av: float = 0, rv: float = 3.1) -> float:
    """Calculate extinction strength.

    See: https://ui.adsabs.harvard.edu/abs/1989ApJ...345..245C/abstract

    :param filter_wavelength: Center wavelength in nanometers.
    :param av: Absolute V-band extinction.
    :param rv: Ratio of total to selective extinction, typically 3.1 for the Milky Way.

    :return: Extinction in magnitudes.
    """
    if av <= 0:
        return 0.0

    x = 1000/filter_wavelength
    y = x - 1.82

    if 0.3 <= x < 1.1:
        # IR
        a = 0.574*x**1.61
        b = -0.527*x**1.61
    elif 1.1 <= x < 3.3:
        # Optical/NIR
        a = 1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 + 0.01979*y**5 - 0.7753*y**6 + 0.32999*y**7
        b = 1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.3026*y**6 - 2.09002*y**7
    elif 3.3 <= x < 8:
        # Near UV
        if x >= 5.9:
            fa = -0.04473*(x - 5.9)**2 - 0.009779*(x - 5.9)**3
            fb = 0.2130*(x - 5.9)**2 + 0.1207*(x - 5.9)**3
        else:
            fa = fb = 0
        a = 1.752 - 0.316*x - 0.104/((x - 4.67)**2 + 0.341) + fa
        b = -3.090 + 1.825*x + 1.206/((x - 4.62)**2 + 0.263) + fb
    elif 8 <= x <= 10:
        # Far UV
        a = -1.073 - 0.628*(x - 8) + 0.137*(x - 8)**2 - 0.070*(x - 8)**3
        b = 13.670 + 4.257*(x - 8) - 0.420*(x - 8)**2 + 0.374*(x - 8)**3
    else:
        a = b = 0

    return av*(a + b/rv)
