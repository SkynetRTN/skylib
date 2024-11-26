
import os

from scipy.interpolate import CubicHermiteSpline, CubicSpline
from astropy import units as u
from astropy.coordinates import CartesianRepresentation, CartesianDifferential, SkyCoord, solar_system_ephemeris
from astropy.time import Time
from skyfield.api import Loader
from skyfield.keplerlib import _KeplerOrbit
from skyfield.data.gravitational_parameters import GM_dict
from skyfield.sgp4lib import EarthSatellite


__all__ = [
    'interpolate_ephem',
    'get_orbital_pos',
    'get_comet_pos',
    'get_planetary_satellite_pos',
    'get_norad_satellite_pos',
    'load_skyfield_data',
]


solar_system_ephemeris.set('jpl')  # for Pluto support in Astropy

skyfield_data_dir = os.environ.get('SKYFIELD_DATA', os.path.expanduser('~/.cache/skyfield'))
load_skyfield_data = Loader(skyfield_data_dir, verbose=False)
ts = load_skyfield_data.timescale()


def interpolate_ephem(coords: list[SkyCoord], t: Time | None = None) -> SkyCoord:
    """
    Interpolate an ephemeris table at the specific epoch(s).

    Parameters
    ----------
    coords : list of `~astropy.coordinates.SkyCoord`
        Positions and optionally velocities at the ephemeris times; each one must have `obstime` set.
    t : `~astropy.time.Time`, optional
        The epoch(s) at which to interpolate the ephemeris. Defaults to now.

    Returns
    -------
    pos : `~astropy.coordinates.SkyCoord`
        The interpolated position(s) and velocities at the specified epoch(s); scalar or array, same length as `t`.
    """
    if t is None:
        t = Time.now()

    # Extract times in float seconds since the starting epoch
    times = Time([c.obstime for c in coords])
    t0 = times.min()
    dt = (times - t0).sec
    dt_target = (t - t0).sec

    # Convert to Cartesian coordinates and velocities in the same units
    coords = [SkyCoord(c) for c in coords]
    for c in coords:
        # Need to set all obstimes to None to make all frames compatible
        c.obstime = None
    coords = SkyCoord(coords)
    cart_coords = coords.replicate()
    cart_coords.representation_type = 'cartesian'
    xyz_unit = cart_coords.x.unit
    vel_unit = xyz_unit/u.s

    # Interpolate with optional velocities, all unitless
    try:
        interp_x = CubicHermiteSpline(dt, cart_coords.x, cart_coords.v_x.to(vel_unit))
        interp_y = CubicHermiteSpline(dt, cart_coords.y, cart_coords.v_y.to(vel_unit))
        interp_z = CubicHermiteSpline(dt, cart_coords.z, cart_coords.v_z.to(vel_unit))
    except TypeError:
        interp_x = CubicSpline(dt, cart_coords.x)
        interp_y = CubicSpline(dt, cart_coords.y)
        interp_z = CubicSpline(dt, cart_coords.z)

    return SkyCoord(
        CartesianRepresentation(
            interp_x(dt_target)*xyz_unit,
            interp_y(dt_target)*xyz_unit,
            interp_z(dt_target)*xyz_unit,
            differentials=CartesianDifferential(
                interp_x.derivative()(dt_target)*vel_unit,
                interp_y.derivative()(dt_target)*vel_unit,
                interp_z.derivative()(dt_target)*vel_unit
            )
        ),
        frame=coords.frame,
        obstime=t,
        equinox=coords.equinox,
    )


def get_orbital_pos(orbit_type: str,
                    epoch: Time,
                    p: float,
                    e: float,
                    incl: float,
                    node: float,
                    peri: float,
                    m: float,
                    t: Time | None = None) -> SkyCoord:
    """
    Compute the position(s) of an orbiting object at the given epoch(s).

    Parameters
    ----------
    orbit_type : str
        The type of orbit, either "geocentric" or "barycentric".
    epoch : `~astropy.time.Time`
        The epoch at which the orbital elements are defined.
    p : float
        The semilatus rectum in AU.
    e : float
        The eccentricity of the orbit.
    incl : float
        The inclination of the orbit in degrees.
    node : float
        The longitude of the ascending node in degrees.
    peri : float
        The argument of the periapsis in degrees.
    m : float
        The mean anomaly in degrees.
    t : `~astropy.time.Time`, optional
        The epoch(s) at which to compute the position(s). Defaults to now.

    Returns
    -------
    pos : `~astropy.coordinates.SkyCoord`
        The position(s) of the object at the specified epoch(s), same shape as `t`.
    """
    if t is None:
        t = Time.now()

    if orbit_type == "barycentric":
        center = 0
    else:
        center = 399

    # See skyfield.data.mpc; since we don't use this module, Pandas is not required
    c = _KeplerOrbit._from_mean_anomaly(
        p, e, incl, node, peri, m, ts.from_astropy(epoch), GM_dict[center], center
    ).at(ts.from_astropy(t)).to_skycoord()
    c.representation_type = "spherical"
    return c


def get_comet_pos(epoch: Time,
                  pd: float,
                  ecc: float,
                  incl: float,
                  node: float,
                  peri: float,
                  t: Time | None = None) -> SkyCoord:
    """
    Compute the position(s) of a comet at the given epoch(s).

    Parameters
    ----------
    epoch : `~astropy.time.Time`
        The epoch of perihelion.
    pd : float
        The perihelion distance of the comet in AU.
    ecc : float
        The eccentricity of the comet.
    incl : float
        The inclination of the comet in degrees.
    node : float
        The longitude of the ascending node in degrees.
    peri : float
        The argument of the perihelion in degrees.
    t : `~astropy.time.Time`, optional
        The epoch(s) at which to compute the position(s). Defaults to now.

    Returns
    -------
    pos : `~astropy.coordinates.SkyCoord`
        The position(s) of the comet at the specified epoch(s), same shape as `t`.
    """
    if t is None:
        t = Time.now()

    center = 0

    # See skyfield.data.mpc
    c = _KeplerOrbit._from_periapsis(
        pd*(1 + ecc), ecc, incl, node, peri, ts.from_astropy(epoch), GM_dict[center], center
    ).at(ts.from_astropy(t)).to_skycoord()
    c.representation_type = "spherical"
    return c


def get_planetary_satellite_pos(bsp: str, satnum: int, t: Time | None = None) -> SkyCoord:
    """
    Compute the position(s) of a planetary satellite at the given epoch(s).

    Parameters
    ----------
    bsp : str
        The name the binary SPK file (.bsp) containing the satellite ephemeris.
        See `~https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/`.
    satnum : int
        The JPL number of the satellite to compute the position of. E.g. Amalthea (JV) is 505.
        Can be found in `~https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/aa_summaries.txt`, as well
        as the corresponding SPK file name.
    t : `~astropy.time.Time`, optional
        The epoch(s) at which to compute the position(s). Defaults to now.

    Returns
    -------
    pos : `~astropy.coordinates.SkyCoord`
        The position(s) of the satellite at the specified epoch(s), same shape as `t`.
    """
    if t is None:
        t = Time.now()

    kernel = load_skyfield_data(bsp)
    try:
        c = kernel[satnum].at(ts.from_astropy(t)).to_skycoord()
        c.representation_type = "spherical"
        return c
    finally:
        kernel.close()


def get_norad_satellite_pos(tle_or_omm: str | dict[str, int | float | str], t: Time | None = None) -> SkyCoord:
    """
    Compute the position(s) of a NORAD satellite at the given epochs(s) from either two-line elements or OMM (Orbit
    Mean-Elements Message) fields.

    Parameters
    ----------
    tle_or_omm : str or dict
        The Two-Line Elements (TLE) of the satellite or a dictionary of OMM fields and their values.
    t : `~astropy.time.Time`, optional
        The epoch(s) at which to compute the position. Defaults to now.

    Returns
    -------
    pos : `~astropy.coordinates.SkyCoord`
        The position(s) of the satellite at the specified epoch(s), same shape as `t`.
    """
    if t is None:
        t = Time.now()

    if isinstance(tle_or_omm, str):
        sat = EarthSatellite(*tle_or_omm.splitlines()[-2:], ts=ts)
    else:
        sat = EarthSatellite.from_omm(ts, tle_or_omm)
    return sat.at(ts.from_astropy(t)).to_skycoord()
