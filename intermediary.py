"""
Calculate intermediary data products like
specific intensities or F and G
"""
import warnings
import os.path
import subprocess
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
warnings.simplefilter('ignore', category=Warning)

def create_bad_pixel_map(obs, threshold=0):
    """ Create a map of all bad pixels from the given set of observations """
    return np.all(obs <= threshold, axis=0) | np.all(obs >= 1-threshold, axis=0)

def doppler_shift(self, spectrum, wl, vel):
    """ Shift spectrum by velocity vel """
    if not isinstance(vel, np.ndarray):
        vel = np.array([vel])
    c0 = 299792  # speed of light in km/s
    # new shifted wavelength grid
    doppler = 1 - vel / c0
    wl_doppler = wl[None, :] * doppler[:, None]
    return np.interp(wl_doppler, wl, spectrum)
    # return interp1d(wl, spectrum, kind=self.config['interpolation_method'], fill_value=0, bounds_error=False)(wl_doppler)

def rv_star(self):
    """ linearly distribute radial velocities during transit """
    return self.config['radial_velocity']

def rv_planet(self, phases):
    """ calculate radial velocities of the planet along the orbit """
    # Orbital speed
    v_orbit = self.par['sma'] * \
        np.sin(self.par['inc']) * 2 * np.pi / self.par['period_s']
    # Modulate with phase
    return v_orbit * np.sin(phases)

def fit_tellurics(self, verbose=False):
    """ fit tellurics using molecfit """
    mfit = os.path.join(
        self.config['intermediary_dir'], self.config['file_molecfit'])
    molecfit = os.path.expanduser(self.config['path_molecfit'])
    sp = subprocess.Popen([molecfit, mfit], stdout=subprocess.PIPE)
    if verbose:
        for line in iter(sp.stdout.readline, ''):
            print(line.decode('utf-8').rstrip())
            if line.decode('utf-8') == '':
                break
        sp.stdout.close()
    sp.wait()

def interpolate_intensity(mu, i):
    """ interpolate the stellar intensity for given limb distance mu """
    return interp1d(i.keys().values, i.values, kind='quadratic', fill_value=0, bounds_error=False, copy=False)(mu).swapaxes(0, 1)

def calc_mu(par, phase):
    """ calculate the distance from the center of the planet to the center of the star as seen from earth """
    """
    distance = self.par['sma'] / self.par['r_star'] * \
        np.sqrt(np.cos(self.par['inc'])**2 +
                np.sin(self.par['inc'])**2 * np.sin(phase)**2)
    """
    return np.sqrt(1 - (par['sma'] / par['r_star'])**2 * (np.cos(par['inc'])**2 + np.sin(par['inc'])**2 * np.sin(phase)**2))

def calc_intensity(par, phase, intensity, min_radius, max_radius, n_radii, n_angle, spacing='equidistant'):
    """
    Calculate the average specific intensity in a given radius range around the planet center
    phase: Phase (in radians) of the planetary transit, with 0 at transit center
    intensity: specific intensity profile of the host star, a pandas DataFrame with keys from 0.0 to 1.0
    min_radius: minimum radius (in km) to sample
    max_radius: maximum radius (in km) to sample
    n_radii: number of radius points to sample
    n_angle: number of angles to sample
    spacing: how to space the samples, 'equidistant' means linear spacing between points, 'random' places them at random positions
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
    # Step 2: Calculate d_x and d_y, distances from the stellar center
    d_x = par['sma'] / par['r_star'] * \
        np.sin(par['inc']) * np.sin(phase)
    d_x = d_x[:, None, None] + \
        (radii[:, None] * np.cos(angles)[None, :])[None, :, :]
    d_y = par['sma'] / par['r_star'] * \
        np.cos(par['inc']) + radii[:, None] * np.sin(angles)[None, :]
    # mu = sqrt(1 - d**2)
    mu = np.sqrt(1 - (d_x**2 + d_y[None, :, :]**2))
    mu = np.nan_to_num(mu, copy=False) #TODO where to do this
    # Step 3: Average specific intensity, outer points weight more, as the area is larger
    intens = interpolate_intensity(mu, intensity)
    intens = np.average(intens, axis=3)
    intens = np.average(intens, axis=2, weights=radii)
    return intens

def maximum_phase(self):
    """ The maximum phase for which the planet is still completely inside the stellar disk """
    # This is the inverse of calc_mu(maximum_phase()) = 1.0
    return np.arcsin(np.sqrt(((self.par['r_star'] - self.par['r_planet'] - self.par['h_atm']) / (
        self.par['sma'] * np.sin(self.par['inc'])))**2 - np.tan(self.par['inc'])**-2))

def specific_intensities(par, phase, intensity, n_radii=11, n_angle=7, mode='precise'):
    """
    Calculate the specific intensities of the star covered by planet and atmosphere, and only atmosphere respectively,
    over the different phases of transit
    phase: phases (in radians) of the transit, with 0 at transit center
    intensity: specific intensity profile of the host star, a pandas DataFrame with keys from 0.0 to 1.0
    n_radii: number of radii to sample, if tuple use n_radii[0] for i_planet and n_radii[1] for i_atm
    n_angle: number of angles to sample, if tuple use n_angle[0] for i_planet and n_angle[1] for i_atm
    mode: fast or precise, fast ignores the planetary disk and only uses the center of the planet, precise uses sample positions inside the radii to determine the average intensity
    """
    # Allow user to specify different n_radii and n_angle for i_planet and i_atm
    if isinstance(n_radii, (float, int)):
        n_radii = (n_radii, n_radii)
    if isinstance(n_angle, (float, int)):
        n_angle = (n_angle, n_angle)

    if mode == 'precise':
        # from r=0 to r = r_planet + r_atmosphere
        i_planet = calc_intensity(par, 
            phase, intensity, 0, (par['r_planet'] + par['h_atm']) / par['r_star'], n_radii[0], n_angle[0])
        # from r=r_planet to r=r_planet+r_atmosphere
        i_atm = calc_intensity(par, 
            phase, intensity, par['r_planet'] / par['r_star'], (par['r_planet'] +par['h_atm']) / par['r_star'], n_radii[1], n_angle[1])
        return i_planet, i_atm
    if mode == 'fast':
        # Alternative version that only uses the center of the planet
        # Faster but less precise (significantly?)
        mu = calc_mu(par, phase)
        intensity = interpolate_intensity(mu, intensity)
        return intensity, intensity

def fit_continuum_alt3(self, wl, spectrum, out='spectrum', size=100, threshhold=4, smoothing=0, plot=False):
    i, j, k = -size, 0, -1
    sparse = np.ones(len(wl)//size + 1, dtype=int)
    while True:
        i += size
        j += size
        k += 1

        sparse[k] = np.argmax(spectrum[i:j]) + i

        if j >= len(wl):
            break

    # Remove Outliers
    for i in range(3):
        diff = np.abs(np.diff(spectrum[sparse]) + np.diff(spectrum[sparse][::-1]))
        sparse = np.delete(sparse, np.where(diff > threshhold * np.median(diff))[0])

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

def fit_continuum_alt2(self, wl, spectrum, out='spectrum'):
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

def fit_continuum_alt(self, wl, spectrum, threshhold=0.001, out='spectrum'):
    if len(spectrum.shape) > 1:
        return np.array([self.fit_continuum_alt(wl, spectrum[i, :], threshhold) for i in range(spectrum.shape[0])])
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

def fit_continuum(self, wl, spectrum, degree=5, percent=10, inplace=True, plot=False, out='spectrum'):
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
