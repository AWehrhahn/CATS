# cython: profile=True

"""
Calculate intermediary data products like
specific intensities or F and G
"""
import warnings

import jdcal
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.integrate import quad, trapz, simps, tplquad

import quadpy

from dataset import dataset

warnings.simplefilter('ignore', category=Warning)


class orbit:
    """Calculates the orbital parameters of the transiting planet

    Functions
    -------
    get_phase(obs_time): {phase}

    get_radius(phase): {radius}

    get_pos(phase): {x, y, z coordinates}

    get_mu(x, y, z, angles, radii): {mu}

    maximum_phase(): {phase}
        gets the maximum phase that is still considered in transit
    rv_planet(phase): {radial velocity}

    plot(x, y, z, mode='2D'): {}
    """

    def __init__(self, par):
        self.par = par

    def get_phase(self, obs_time):
        """Calculate the orbit phase depending on the obs_time

        (time - (trasit-period/2)) / period modulo 1

        Parameters:
        ----------
        par : {dict}
            dict of orbit parameters
        obs_time : {float, np.ndarray}
            observation times in MJD
        Returns
        -------
        phase: {float, np.ndarray}
            orbital phase in degrees
        """

        transit = self.par['transit'] - jdcal.MJD_0
        period = self.par['period']
        phase = ((obs_time - (transit - period / 2)) / period) % 1
        phase = 2 * np.pi * phase  # in rad
        return phase

    def get_radius(self, phase):
        """calculate the radius at the given phase

        this is NOT the radius in the direction of observation, but the radius in polar coordinates

        Parameters:
        ----------
        phase : {float, np.ndarray}
            orbital phase in radians
        Returns
        -------
        radius : {float, np.ndarray}
            radius in units of stellar radii
        """

        phi = phase - self.get_phase(self.par['periastron'])
        a = self.par['sma']
        e = self.par['eccentricity']
        radius = a * (1 - e**2) / (1 + e * np.cos(phi))
        return radius / self.par['r_star']

    def get_pos(self, phase):
        """Calculate the 3D position of the planet

        the coordinate system is centered in the star, x is towards the observer, z is orthagonal to the planet orbit, and y to the "right"

          z ^
            | 
            | -¤-
            |̣_____>
            /      y
           / x

        Parameters:
        ----------
        phase : {float, np.ndarray}
            phase in radians
        Returns
        -------
        x, y, z: {float, np.ndarray}
            position in stellar radii
        """

        r = self.get_radius(phase)
        i = self.par['inc']
        x = -r * np.cos(phase) * np.sin(i)
        y = -r * np.sin(phase)
        z = -r * np.cos(phase) * np.cos(i)
        return x, y, z

    def maximum_phase(self):
        """ The maximum phase for which the planet is still completely inside the stellar disk

        based only on geometry
        This is the inverse of calc_mu(maximum_phase()) = 0.0, if there is no inclination

        Parameters:
        ----------
        par : {dict}
            stellar and planetary parameters
        Returns
        -------
        phase : float
            maximum phase (in radians)
        """

        # func == y**2 + z**2 - 1 = 0
        max_phase = fsolve(lambda phase: np.sum(
            np.power(self.get_pos(phase)[1:], 2)) - 1, 3.14)
        return max_phase

    def get_mu(self, x, y, z, angles=None, radii=None):
        """get mu = cos(distance to stellar center)
        cos(limb distance), where 1 is the center of the star and 0 is the outer edge

        project onto yz-plane, i.e. ignore x and calculate the distance

        Parameters:
        ----------
        x : {float, np.ndarray}
            x position in stellar radii
        y : {float, np.ndarray}
            y position in stellar radii
        z : {float, np.ndarray}
            z position in stellar radii
        Returns
        -------
        mu : float, np.ndarray
            cos(sqrt(y**2 + z**2))
        """
        if not isinstance(y, np.ndarray):
            y = np.array([y])
        if not isinstance(z, np.ndarray):
            z = np.array([z])

        if radii is not None and angles is not None:
            y = y[:, None, None] + np.outer(radii, np.cos(angles))[None, ...]
            z = z[:, None, None] + np.outer(radii, np.sin(angles))[None, ...]

        mu = np.sqrt(1 - (y**2 + z**2))
        mu[np.isnan(mu)] = -1
        return mu

    def rv_planet(self, phase):
        """ Calculate the radial velocity of the planet at a given phase

        Uses only simple geometry

        Parameters:
        ----------
        par : {dict}
            stellar and planetary parameters
        phases : {float, np.ndarray}
            orbital phases of the planet
        Returns
        -------
        rv : {float, np.ndarray}
            radial velocity of the planet in km/s
        """

        """ calculate radial velocities of the planet along the orbit """
        # radius
        r = self.get_radius(phase) * self.par['r_star']  # km
        a = self.par['sma']  # km
        i = self.par['inc']  # rad
        # standard gravitational parameter
        sgp = scipy.constants.gravitational_constant * \
            self.par['m_star'] * 1e-9  # km**3/s**2

        # calculate orbital velocity
        v = np.sqrt(sgp * (2 / r - 1 / a))

        # Get line of sight component
        v *= np.sin(phase) * np.sin(i)

        return v

    def plot(self, x, y, z, mode='2D'):
        """Plot the star together with planet at position x, y, z

        Parameters:
        ----------
        x : {np.ndarray}
            x-coordinate
        y : {np.ndarray}
            y-coordinate
        z : {np.ndarray}
            z-coordinate
        """

        if mode == '2D':
            c_star = plt.Circle((0, 0), 1, color='y')

            r = (self.par['r_planet'] + self.par['h_atm']) / self.par['r_star']
            c = [plt.Circle((j, k), r, color='b', alpha=0.5)
                 for i, j, k in zip(x, y, z)]

            fig, ax = plt.subplots()
            ax.add_artist(c_star)
            ax.plot(y, z, 'o', color='r')
            for circle in c:
                ax.add_artist(circle)

            plt.xlabel('y')
            plt.ylabel('z')
            plt.xlim((-1.5, 1.5))
            plt.ylim((-1.5, 1.5))
            plt.show()

        if mode == '3D':
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_proj_type('ortho')

            # draw sphere/star
            def plot_sphere(x, y, z, radius, color='b', alpha=0.5):
                u = np.linspace(0, 2 * np.pi, 100)
                v = np.linspace(0, np.pi, 100)

                _x = x + radius * np.outer(np.cos(u), np.sin(v))
                _y = y + radius * np.outer(np.sin(u), np.sin(v))
                _z = z + radius * np.outer(np.ones(np.size(u)), np.cos(v))
                ax.plot_surface(_x, _y, _z, rstride=4, cstride=4,
                                color=color, linewidth=0, alpha=alpha)

            plot_sphere(0, 0, 0, 1, 'y', 1)

            r = (self.par['r_planet'] + self.par['h_atm']) / self.par['r_star']
            for i, j, k in zip(x, y, z):
                ax.scatter(i, j, k, color='r')
                plot_sphere(i, j, k, r, 'b')

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.view_init(elev=0, azim=0)

            plt.show()


def rv_planet(par, phases):
    """ Calculate the radial velocity of the planet at a given phase

    Uses only simple geometry

    Parameters:
    ----------
    par : {dict}
        stellar and planetary parameters
    phases : {float, np.ndarray}
        orbital phases of the planet
    Returns
    -------
    rv : {float, np.ndarray}
        radial velocity of the planet in km/s
    """
    orb = orbit(par)
    v = orb.rv_planet(phases)
    return v


def maximum_phase(par):
    """ The maximum phase for which the planet is still completely inside the stellar disk

    based only on geometry
    This is the inverse of calc_mu(maximum_phase()) = 0.0, if there is no inclination

    Parameters:
    ----------
    par : {dict}
        stellar and planetary parameters
    Returns
    -------
    phase : float
        maximum phase (in radians)
    """
    orb = orbit(par)
    return orb.maximum_phase()


def calc_mu(par, phase, angles=None, radii=None):
    """calculate the distance from the center of the planet to the center of the star as seen from earth

    Parameters:
    ----------
    par : {dict}
        stellar and planetary parameters
    phase : {float, np.ndarray}
        orbital phase of the planet
    angles: {np.ndarray}, optional
        set of angles to sample around the center of the planet (in radians)
    radii: {np.ndarray}, optional
        set of radii to sample around the center of the planet (in km)
    Returns
    -------
    mu : {float, np.ndarray}
        cos(limb distance), where 0 is the center of the star and 1 is the outer edge
    """

    orb = orbit(par)
    pos = orb.get_pos(phase)
    return orb.get_mu(*pos, angles=angles, radii=radii)


def create_bad_pixel_map(obs, threshold=0):
    """ Create a map of all bad pixels from the given set of observations

    Parameters:
    ----------
    obs : {dataset}
        observations
    threshold : {float}, optional
        determines how close a pixel has to be to 0 or 1 to be considered a bad pixel (the default is 0, which is exact)
    Returns
    -------
    badpixelmap : np.ndarray
        Bad pixel map, same dimensions as obs.flux - 1
    """
    return np.all(np.gradient(obs.flux) == 0, axis=0) | np.all(obs.flux <= threshold, axis=0) | np.all(obs.flux >= np.max(obs.flux) - threshold, axis=0)


def profile_exponential(par, h):
    scale_height = par['atm_scale_height'] / par['r_star'] * 10000
    return np.exp(-h / scale_height)


def profile_solid(par, h):
    r = np.min(h, axis=1)
    ha = np.max(h)
    L = np.sqrt(ha**2 - r**2)
    return np.repeat(0.5 / L, h.shape[1]).reshape(h.shape)


def atmosphere_profile(par, phase, intensity, r0, h, profile=profile_exponential, n_radii=11, n_angle=12, n_depth=13):
    orb = orbit(par)
    _, y, z = orb.get_pos(phase)

    r = np.linspace(r0, h, n_radii, endpoint=False)
    theta = np.linspace(0, 2 * np.pi, n_angle, endpoint=False)
    L = np.sqrt(h**2 - r**2)
    x = np.array([np.linspace(0, l, n_depth) for l in L])
    x_dash = np.sqrt(r[:, None]**2 + x**2)

    rho_i = 2 * simps(profile(par, x_dash - r0), x, axis=1)
    rho_i = np.nan_to_num(rho_i, copy=False)

    mu = orb.get_mu(_, y, z, angles=theta, radii=r)
    I = interpolate_intensity(mu, intensity)

    # TODO optimize this multiplication?
    tmp = simps(I * mu[..., None] * rho_i[None, :, None, None]
                * r[None, :, None, None], r, axis=1)
    res = simps(tmp, theta, axis=1) / np.pi  # TODO normalize to area of star
    return res


def test_profile(par, phase, intensity, r0, *args, **kwargs):
    orb = orbit(par)
    _, y, z = orb.get_pos(phase)
    H = par['atm_scale_height']
    r0 = r0 * par['r_star']

    mu = lambda r, theta: orb.get_mu(0, y, z, angles=theta, radii=r)
    I = lambda r, theta: interpolate_intensity(orb.get_mu(0, y, z, angles=theta, radii=r), intensity)
    rho = lambda r, x: np.exp(-(np.sqrt(r**2-x**2)-r0)/H)

    func = lambda x, theta, r:  mu(r, theta)[0] * I(r, theta)[0, 0] * rho(r, x) * r
    res = tplquad(func, r0, np.inf, lambda x: 0, lambda x: 2*np.pi, lambda x, theta: -np.inf, lambda x, theta: +np.inf)
    return res

def interpolate_intensity(mu, i):
    """ Interpolate the stellar intensity for given limb distance mu

    use linear interpolation, because it is much faster

    Parameters:
    ----------
    mu : {float, np.ndarray}
        cos(limb distance), i.e. 1 is the center of the star, 0 is the outer edge
    i : {pd.DataFrame}
        specific intensities
    Returns
    -------
    intensity : np.ndarray
        interpolated intensity
    """
    # TODO can I optimize this?
    values = i.values.swapaxes(0, 1)
    return interp1d(i.keys(), values, kind='zero', axis=0, bounds_error=False, fill_value=(0, values[1]))(mu)


def calc_intensity(par, phase, intensity, min_radius, max_radius, n_radii, n_angle, spacing='equidistant'):
    """Calculate the average specific intensity in a given radius range around the planet center

    Parameters:
    ----------
    par : {dict}
        stellar and planetary parameters
    phase : {float, np.ndarray}
        orbital phase of the planet (in radians)
    intensity : {datset}
        specific intensities
    min_radius : {float}
        minimum radius from the planet center to use for calculations (in km)
    max_radius : {float}
        maximum radius from planet center to use for calculations (im km)
    n_radii : {int}
        number of radii to sample
    n_angle : {int}
        number of angles to sample
    spacing : {'equidistant', 'random'}
        wether to use equidistant sampling points or a random distribution (default is 'equidistant')
    Returns
    -------
    intensity : np.ndarray
        specific intensities
    """
    # Step 1: Calculate sampling positions in the given radii
    if spacing in ['e', 'equidistant']:
        # equidistant spacing
        radii = np.linspace(min_radius, max_radius, n_radii, endpoint=True)
        # No endpoint means no overlap -> no preference (but really thats just a small difference)
        angles = np.linspace(0, 2 * np.pi, n_angle, endpoint=False)
    if spacing in ['r', 'random', 'mc']:
        # random spacing (Monte-Carlo)
        radii = np.random.random_sample(
            n_radii) * (max_radius - min_radius) + min_radius
        angles = np.random.random_sample(n_angle) * 2 * np.pi

    # Step 2: Calculate mu, distances from the stellar center
    radii /= par['r_star']
    mu = calc_mu(par, phase, angles=angles, radii=radii)
    # -1 is out of bounds, which will be filled with 0 intensity
    mu[np.isnan(mu)] = -1

    # Step 3: Average specific intensity, outer points weight more, as the area is larger
    # flux = integrate intensity(mu) * mu dmu = Area * average(Intensity * mu)
    intens = interpolate_intensity(mu, intensity)
    intens = np.average(intens, axis=2)
    intens = np.average(intens, axis=1, weights=radii)
    return intens


def specific_intensities(par, phase, intensity, n_radii=11, n_angle=7, mode='fast'):
    """Calculate the specific intensities of the star covered by planet and atmosphere, and only atmosphere respectively,
    over the different phases of transit

    TODO: are the precise calculation worth it ???
    TODO: Allow user to specify different n_radii and n_angle for i_planet and i_atm

    Parameters:
    ----------
    par : {dict}
        stellar and planetary parameters
    phase : {float, np.ndarray}
        orbital phase in radians
    intensity : {dataset}
        specific intensities
    n_radii : {int}, optional
        number of radii to sample (the default is 11)
    n_angle : {int}, optional
        number of angles to sample (the default is 7)
    mode : {'precise', 'fast'}
        in precise mode, various radii and angle in the atmosphere and body of the planet are sampled
        inb fast mode, use only the center of the planet
    Returns
    -------
    i_planet, i_atm : dataset, dataset
        specific intensities blocked by the planet body and the atmosphere
    """

    if isinstance(n_radii, (float, int)):
        n_radii = (n_radii, n_radii)
    if isinstance(n_angle, (float, int)):
        n_angle = (n_angle, n_angle)

    if mode == 'profile':
        r0 = par['r_planet'] / par['r_star']
        h = r0 + par['h_atm'] / par['r_star']

        planet = atmosphere_profile(
            par, phase, intensity.flux, 0, h, profile_solid, n_radii[0], n_angle[0])
        atm = atmosphere_profile(
            par, phase, intensity.flux, r0, h, profile_exponential, n_radii[1], n_angle[1])

        ds_planet = dataset(intensity.wl, planet)
        ds_atm = dataset(intensity.wl, atm)
        return ds_planet, ds_atm

    if mode == 'precise':
        # from r=0 to r = r_planet + r_atmosphere
        inner = 0
        outer = par['r_planet'] + par['h_atm']
        i_planet = calc_intensity(
            par, phase, intensity.flux, inner, outer, n_radii[0], n_angle[0])

        # from r=r_planet to r=r_planet+r_atmosphere
        inner = par['r_planet']
        outer = par['r_planet'] + par['h_atm']
        i_atm = calc_intensity(par, phase, intensity.flux,
                               inner, outer, n_radii[1], n_angle[1])
        ds_planet = dataset(intensity.wl, i_planet)
        ds_atm = dataset(intensity.wl, i_atm)
        return ds_planet, ds_atm

    if mode == 'fast':
        # Alternative version that only uses the center of the planet
        # Faster but less precise (significantly?)
        mu = calc_mu(par, phase)
        _flux = interpolate_intensity(mu, intensity.flux)
        ds = dataset(intensity.wl, np.copy(_flux))
        ds2 = dataset(intensity.wl, np.copy(_flux))
        return ds, ds2
