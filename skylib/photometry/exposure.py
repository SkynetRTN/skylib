"""
Exposure time calculator based on the sky brightness and the desired signal-to-noise ratio.
"""

import warnings

import numpy as np
from numba import njit
from scipy.optimize import fsolve, least_squares
from astropy.time import Time
from astropy.coordinates import AltAz, EarthLocation, GCRS, NonRotationTransformationWarning, SkyCoord, get_body

from ..util.angle import airmass_for_el


__all__ = [
    "moon_brightness", "sky_brightness", "calibrate_sky_model",
    "exptime_for_mag_and_snr", "mag_for_exptime_and_snr", "snr_for_mag_and_exptime", "flux15_for_exptime_mag_and_snr",
]


def moon_brightness(
        t: Time,
        site: EarthLocation,
        target: SkyCoord,
        sun: SkyCoord | None = None,
        m0: float = 18.0,
        kr: float = 1.06,
        krm: float = 4.1,
        g: float = 0.8,
        tau: float = 0.138,
) -> float:
    """
    Estimate the Moon's contribution to the total sky brightness following Winkler, H. "A revised simplified scattering
    model for the moonlit sky brightness profile based on photometry at SAAO" // 2022, MNRAS, 514(1), 208--226

    Default values are for the V band, taken from Winkler (2022).

    :param t: Observation time.
    :param site: Observation site.
    :param target: Target coordinates.
    :param sun: Sun coordinates. If None, the Sun's coordinates will be calculated.
    :param m0: Moon brightness zero point for the given bandpass in magnitudes per arcsecond squared.
    :param kr: Moonlight Rayleigh scattering coefficient = (1 + 3 chi)/(1 - chi) as per Bucholtz (1995).
    :param krm: Ratio of moonlight Rayleigh to aerosol scattering optical depths (> 0).
    :param g: Moonlight aerosol scattering asymmetry parameter in the Henyey--Greenstein model (> 0 and < 1).
    :param tau: Total moonlight optical depth = 0.921 k, where k is the atmospheric extinction coefficient (> 0).

    :returns: Estimated Moon's brightness contribution at the location of the target, in magnitudes per arcsecond
        squared.
    """
    # Moon airmass
    moon = get_body("moon", t, site)
    moon_elevation = moon.transform_to(AltAz(obstime=t, location=site)).alt.deg
    if moon_elevation <= 0:
        # No moonlight contribution
        return np.inf
    sec_z = airmass_for_el(moon_elevation)

    # Moon phase angle
    if sun is None:
        sun = get_body("sun", t, site)
    phase_angle = moon.separation(sun).rad
    if not phase_angle:
        # Solar eclipse
        return np.inf

    # Target airmass and Moon separation (need to convert to topocentric first)
    obsgeoloc, obsgeovel = site.get_gcrs_posvel(t)
    local_target_coord = SkyCoord(target, location=site, obstime=t).transform_to(GCRS(
        obstime=t, obsgeoloc=obsgeoloc, obsgeovel=obsgeovel, representation_type='spherical'
    ))
    sec_zeta = airmass_for_el(local_target_coord.transform_to(AltAz(obstime=t, location=site)).alt.deg)
    cos_theta = np.cos(local_target_coord.separation(moon).rad)

    att = (1 - g**2)/(1 + g**2 - 2*g*cos_theta)**1.5 + krm*(kr + cos_theta**2)/(kr + 1/3)  # Henyey--Greenstein
    att *= sec_zeta*np.where(
        sec_z == sec_zeta,
        tau*np.exp(-tau*sec_zeta),
        (np.exp(-tau*sec_zeta) - np.exp(-tau*sec_z))/(sec_z - sec_zeta))
    normalized_flux = (
        (1 - np.cos(phase_angle))/2  # phase function
        * att  # attenuation
    )
    if normalized_flux <= 0:
        return np.inf

    return m0 - 2.5*np.log10(normalized_flux)


@njit(nogil=True, cache=True)
def kastner_log_l_1a(z: float, theta: float, h: float) -> float:
    """
    Return logL as per (1a) in Kastner (1976)

    :param z: zenith angle of target in degrees
    :param theta: azimuth separation between sun and target in degrees
    :param h: solar elevation in degrees

    :return: logL for theta < theta0
    """
    return -(7.5e-5*z + 5.05e-3)*theta - (3.67e-4*z - 0.458)*h + 9.17e-3*z + 3.525


@njit(nogil=True, cache=True)
def kastner_log_l_1b(z: float, theta: float, h: float) -> float:
    """
    Return logL as per (1b) in Kastner (1976)

    :param z: zenith angle of target in degrees
    :param theta: azimuth separation between sun and target in degrees
    :param h: solar elevation in degrees

    :return: logL for theta > theta0
    """
    return -0.001*theta - (1.12e-3*z - 0.47)*h - 4.17e-3*z + 3.225


@njit(nogil=True, cache=True)
def kastner_log_l(a: float, z: float, sun_a: float, h: float) -> float:
    """
    Return twilight sky flux as per Kastner (1976) 1976JRASC..70..153K

    :param a: azimuth of target in degrees
    :param z: zenith angle of target in degrees
    :param sun_a: solar azimuth in degrees
    :param h: solar elevation (normally negative) in degrees

    :return: logarithm of twilight flux in arbitrary units; should be normalized by zero point
    """
    if z < 30:
        z = 30
    elif z > 90:
        z = 90
    theta_0 = (4.12e-2*z + 0.582)*h + 0.417*z + 97.5  # (1c)
    theta = (a - sun_a) % 360
    if theta > 180:
        theta = 360 - theta
    if theta <= theta_0:
        log_l_30 = kastner_log_l_1a(30, theta, h)
        log_l_90 = kastner_log_l_1a(90, theta, h)
    else:
        log_l_30 = kastner_log_l_1b(30, theta, h)
        log_l_90 = kastner_log_l_1b(90, theta, h)

    # Correction from Note, p.167
    z_30 = z/30
    return (z_30 - 2)*log_l_90 + (3 - z_30)*log_l_30


# Twilight flux normalization: sky brightness = night sky brightness at the antisolar point when sun elevation = -18
log_l0 = kastner_log_l(180, 18, 0, -18)  # -5.8947


def sky_brightness(
        t: Time,
        site: EarthLocation,
        target: SkyCoord,
        night_sky_brightness: float = 22,
        twilight_coeff: float = 2.5*np.log(10),
        moonlight_zero_point: float = 18.0,
        moonlight_rayleigh_coeff: float = 1.06,
        moonlight_aerosol_coeff: float = 0.8,
        moonlight_rayleigh_to_aerosol_ratio: float = 4.1,
        moonlight_optical_depth: float = 0.138,
) -> float:
    """
    Estimate the sky brightness based on the Sun's elevation, Moon's phase, and the mutual position of Sun, Moon, and
    target.

    :param t: Observation time.
    :param site: Observation site.
    :param target: Target coordinates.
    :param night_sky_brightness: Night sky brightness for the given bandpass in magnitudes per arcsecond squared.
    :param twilight_coeff: Twilight sky brightness growth factor (> 0).
    :param moonlight_zero_point: Moon brightness zero point for the given bandpass in magnitudes per arcsecond squared.
    :param moonlight_rayleigh_coeff: Moonlight Rayleigh scattering coefficient = (1 + 3chi)/(1 - chi) as per Bucholtz
        (1995)
    :param moonlight_aerosol_coeff: Moonlight aerosol scattering asymmetry parameter in the Henyey--Greenstein model
        (> 0 and < 1).
    :param moonlight_rayleigh_to_aerosol_ratio: Ratio of moonlight Rayleigh to aerosol scattering optical depths (> 0).
    :param moonlight_optical_depth: Total moonlight optical depth = 0.921 k, where k is the atmospheric extinction
        coefficient (> 0).

    :returns: Estimated sky brightness in magnitudes per arcsecond squared.
    """
    # Nighttime and twilight sky contribution
    sun = get_body("sun", t, site)
    sun_altaz = sun.transform_to(AltAz(obstime=t, location=site))
    sun_h = sun_altaz.alt.deg
    sky_mag = float(night_sky_brightness)
    if sun_h > -18:
        # Twilight sky brightness from Kastner (1976) 1976JRASC..70..153K; normalized so that the minimum brightness
        # equals the night sky brightness
        target_altaz = target.transform_to(AltAz(obstime=t, location=site))
        log_l = kastner_log_l(target_altaz.az.deg, target_altaz.zen.deg, sun_altaz.az.deg, sun_h)
        sky_mag -= twilight_coeff*(log_l - log_l0)

    # Moonlight contribution
    moon_mag = moon_brightness(
        t, site, target, sun, m0=moonlight_zero_point, kr=moonlight_rayleigh_coeff,
        krm=moonlight_rayleigh_to_aerosol_ratio, g=moonlight_aerosol_coeff, tau=moonlight_optical_depth)
    if np.isfinite(moon_mag):
        # Combine nighttime sky, twilight, and moonlight contributions
        return -2.5*np.log10(10**(-0.4*sky_mag) + 10**(-0.4*moon_mag))

    return sky_mag


def calibrate_sky_model(t: Time, site: EarthLocation, targets: SkyCoord, mags: np.ndarray,
                        night_sky_brightness: float = 22, fix_night_sky_brightness: bool = False,
                        twilight_coeff: float = 0.44, fix_twilight_coeff: bool = False,
                        moonlight_zero_point: float = -12.7, fix_moonlight_zero_point: bool = False,
                        moonlight_aerosol_coeff: float = 0.8, fix_moonlight_aerosol_coeff: bool = False) \
        -> tuple[float, float, float, float]:
    """
    Calculate the three sky background model parameters given a set of measured sky background magnitudes.

    To reliably estimate all model parameters, the input dataset must contain observations with considerably high
    background levels and obtained both with and without the effect of the Moon and both at twilight and full night.

    :param t: array of observation epochs.
    :param site: observation site.
    :param targets: array of target coordinates, same shape as `t`.
    :param mags: array of measured sky background magnitudes, same shape as `t`.
    :param night_sky_brightness: nighttime sky brightness for the given bandpass in magnitudes per arcsecond squared;
        fixed value if `fix_night_sky_brightness` = True or initial guess otherwise
    :param fix_night_sky_brightness: should `night_sky_brightness` be fixed or estimated from the data?
    :param twilight_coeff: twilight sky brightness growth factor (> 0); fixed value if `fix_twilight_coeff` = True
        or initial guess otherwise.
    :param fix_twilight_coeff: should `twilight_coeff` be fixed or estimated from the data?
    :param moonlight_zero_point: Moon brightness zero point for the given bandpass in magnitudes per arcsecond squared.
    :param fix_moonlight_zero_point: should `moonlight_zero_point` be fixed or estimated from the data?
    :param moonlight_aerosol_coeff: moonlight aerosol scattering asymmetry parameter in the Henyey--Greenstein model
        (> 0 and < 1).
    :param fix_moonlight_aerosol_coeff: should `moonlight_aerosol_coeff` be fixed or estimated from the data?

    :returns: sky background model parameters:
        - `night_sky_brightness`: nighttime sky brightness for the given bandpass in magnitudes per arcsecond squared.
        - `twilight_coeff`: twilight sky brightness growth factor.
        - `moonlight_zero_point`: Moon brightness zero point for the given bandpass.
        - `moonlight_attenuation`: moonlight aerosol scattering asymmetry parameter in the Henyey--Greenstein model.
    """
    def func(x):
        # Parse parameters that can vary
        i = 0
        if fix_night_sky_brightness:
            nsb = night_sky_brightness
        else:
            nsb = x[i]
            i += 1
        if fix_twilight_coeff:
            tc = twilight_coeff
        else:
            tc = x[i]
            i += 1
        if fix_moonlight_zero_point:
            mzp = moonlight_zero_point
        else:
            mzp = x[i]
            i += 1
        if fix_moonlight_aerosol_coeff:
            ma = moonlight_aerosol_coeff
        else:
            ma = x[i]
            i += 1

        res = mags.copy()
        for i, (ti, ci) in enumerate(zip(t, targets)):
            res[i] -= sky_brightness(
                ti, site, ci, night_sky_brightness=nsb, twilight_coeff=tc, moonlight_zero_point=mzp,
                moonlight_aerosol_coeff=ma,
            )
        return res

    warnings.filterwarnings('ignore', category=NonRotationTransformationWarning)

    x0, bounds_lo, bounds_hi = [], [], []
    if not fix_night_sky_brightness:
        x0.append(night_sky_brightness)
        bounds_lo.append(10)
        bounds_hi.append(25)
    if not fix_twilight_coeff:
        x0.append(twilight_coeff)
        bounds_lo.append(0)
        bounds_hi.append(1.5)
    if not fix_moonlight_zero_point:
        x0.append(moonlight_zero_point)
        bounds_lo.append(-13)
        bounds_hi.append(-10)
    if not fix_moonlight_aerosol_coeff:
        x0.append(moonlight_aerosol_coeff)
        bounds_lo.append(0)
        bounds_hi.append(1)

    if x0:
        params = least_squares(func, x0, bounds=(bounds_lo, bounds_hi)).x
        p = 0
        if not fix_night_sky_brightness:
            night_sky_brightness = params[p]
            p += 1
        if not fix_twilight_coeff:
            twilight_coeff = params[p]
            p += 1
        if not fix_moonlight_zero_point:
            moonlight_zero_point = params[p]
            p += 1
        if not fix_moonlight_aerosol_coeff:
            moonlight_aerosol_coeff = params[p]

    return night_sky_brightness, twilight_coeff, moonlight_zero_point, moonlight_aerosol_coeff


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
