# cython: profile=True

"""
Calculate intermediary data products like
specific intensities or F and G
"""
import warnings

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import scipy.constants

from dataset import dataset

import jdcal

warnings.simplefilter('ignore', category=Warning)

# TODO include eccentricity in orbit solution


class orbit:
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
        if radii is not None and angles is not None:
            y = y[:, None, None] + radii[None, :, None] * \
                np.cos(angles)[None, None, :]
            z = z[:, None, None] + radii[None, :, None] * \
                np.sin(angles)[None, None, :]

        mu = np.sqrt(1 - (y**2 + z**2))
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
                ax.plot_surface(_x, _y, _z,  rstride=4, cstride=4,
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

    # orb.plot(*pos)

    return orb.get_mu(*pos, angles=angles, radii=radii)


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
    intens = interpolate_intensity(mu, intensity) * mu[..., None]
    intens = np.average(intens, axis=2)
    intens = np.average(intens, axis=1, weights=radii)
    return intens


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


def specific_intensities(par, phase, intensity, n_radii=11, n_angle=7, mode='precise'):
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
        ds = dataset(intensity.wl, _flux)
        return ds, ds


def fit_continuum_alt3(wl, spectrum, out='spectrum', size=100, threshhold=4, smoothing=0, plot=False):
    i, j, k = -size, 0, -1
    sparse = np.ones(len(wl) // size + 1, dtype=int)
    while True:
        i += size
        j += size
        k += 1

        sparse[k] = np.argmax(spectrum[i:j]) + i

        if j >= len(wl):
            break

    # Remove Outliers
    for i in range(3):
        diff = np.abs(np.diff(spectrum[sparse]) +
                      np.diff(spectrum[sparse][::-1]))
        sparse = np.delete(sparse, np.where(
            diff > threshhold * np.median(diff))[0])

    fit = np.interp(wl, wl[sparse], spectrum[sparse])
    #poly = UnivariateSpline(wl[sparse], spectrum[sparse], s=smoothing, ext=3)
    #fit = poly(wl)

    if plot:
        plt.plot(wl, spectrum)
        plt.plot(wl, fit)
        plt.plot(wl[sparse], spectrum[sparse], 'd')
        plt.show()

    if out == 'norm':
        return fit
    if out == 'spectrum':
        return spectrum / fit


def fit_continuum_alt2(wl, spectrum, out='spectrum'):
    j = np.argmax(spectrum)
    mask = np.zeros(len(wl), dtype=bool)
    mask[j] = True
    while j < len(wl) - 1:
        # + (wl[j]/100 - wl[j+1:]/100)**2
        distance = (spectrum[j] - spectrum[j + 1:])**2
        shorty = np.argmin(distance)
        j += shorty + 1
        mask[j] = True

    j = np.argmax(spectrum)
    while j > 1:
        distance = (spectrum[j] - spectrum[:j - 1])**2
        shorty = np.argmin(distance)
        j -= shorty + 1
        mask[j] = True

    norm = interp1d(wl[mask], spectrum[mask],
                    fill_value='extrapolate', kind='slinear')(wl)

    plt.plot(wl, spectrum, wl, norm)
    plt.show()

    if out == 'spectrum':
        return spectrum / norm
    if out == 'norm':
        return norm


def fit_continuum_alt(wl, spectrum, threshhold=0.001, out='spectrum'):
    if len(spectrum.shape) > 1:
        return np.array([fit_continuum_alt(wl, spectrum[i, :], threshhold) for i in range(spectrum.shape[0])])
    prime = np.gradient(spectrum, wl)
    second = np.gradient(spectrum, wl)

    mask = (abs(prime) <= threshhold) & (second <= 0)
    spec_new = interp1d(wl[mask], spectrum[mask],
                        kind='linear', fill_value='extrapolate')(wl)

    prime = np.gradient(spec_new, wl)
    second = np.gradient(spec_new, wl)

    mask = (abs(prime) <= threshhold * 0.1) & (second <= 0)

    count = len(np.where(mask)[0])
    print(count)
    norm = np.polyfit(wl[mask], spec_new[mask], 7)
    norm = np.polyval(norm, wl)

    if out == 'spectrum':
        return spectrum / norm
    if out == 'norm':
        return norm


def fit_continuum(wl, spectrum, degree=5, percent=10, inplace=True, plot=False, out='spectrum'):
    """ fit a continuum to the spectrum and continuum normalize it """
    def fit_polynomial(wl, spectrum, mask, percent):
        poly = np.polyfit(wl[mask], spectrum[mask], degree)
        fit = np.polyval(poly, wl)
        mask = spectrum >= fit

        # Add the x percen smallest difference points back to the fit
        sort = np.argsort(
            np.abs(spectrum[~mask] - fit[~mask]))[:len(spectrum[~mask]) // percent]
        mask[np.where(~mask)[0][sort]] = True
        percent += 1

        if plot:
            plt.plot(wl, spectrum, wl, fit)
            plt.plot(wl[mask], spectrum[mask], ',')
            plt.show()
        return mask, fit, percent

    if not inplace:
        spectrum = np.copy(spectrum)

    mask = np.ones(spectrum.shape, dtype=bool)
    fit = np.empty(spectrum.shape)
    while True:
        if len(spectrum.shape) == 1:
            mask, fit, percent = fit_polynomial(
                wl, spectrum, mask, percent)
            count = len(spectrum[spectrum <= fit])
        else:
            if isinstance(spectrum, pd.DataFrame):
                for i in range(spectrum.shape[1]):
                    mask[:, i], fit[:, i], percent = fit_polynomial(
                        wl, spectrum.iloc[:, i], mask[:, i], percent)
                count = spectrum[spectrum <= fit].count().sum()

            else:
                for i in range(spectrum.shape[0]):
                    mask[i], fit[i], percent = fit_polynomial(
                        wl, spectrum[i], mask[i], percent)
                count = np.product(spectrum[spectrum <= fit].shape)

        # if 99% are lower than the fit, thats a good continuum ?
        if count >= 0.99 * np.product(spectrum.shape):
            break

    if plot:
        if len(spectrum.shape) > 1:
            plt.plot(wl, spectrum[0], wl, fit[0])
        else:
            plt.plot(wl, spectrum, wl, fit)
        plt.title('Final')
        plt.show()

    if out == 'spectrum':
        return spectrum / fit
    if out == 'norm':
        return fit

    raise AttributeError('value of out parameter unknown')
