"""
Use an inverse method approach to determine the planetary transit spectrum
author: Ansgar Wehrhahn (ansgar.wehrhahn@physics.uu.se)
Based on work by Nikolai Piskunov (Uppsala University)
"""

import os.path
import numpy as np

from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d as gaussbroad, maximum_filter1d
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

from read_write import read_write
from intermediary import intermediary
from solution import solution


def load_keck_save(filename):
    """ just a reminder how to load the keck info file """
    import scipy.io
    import pandas as pd
    keck = scipy.io.readsav(filename)
    cat = keck['cat']
    df = pd.DataFrame.from_records(cat)
    df.applymap(lambda s: s.decode('ascii') if isinstance(s, bytes) else s)
    return df


def rebin(a, newshape):
    '''
    Rebin an array to a new shape.
    '''
    assert len(a.shape) == len(newshape)

    slices = [slice(0, old, float(old) / new)
              for old, new in zip(a.shape, newshape)]
    coordinates = np.mgrid[slices]
    # choose the biggest smaller integer index
    indices = coordinates.astype('i')
    return a[tuple(indices)]


def normalize2d(arr, axis=1):
    """ normalize array arr """
    # TODO fix dimensionality
    arr -= np.min(arr, axis=axis)[:, None]
    arr /= np.max(arr, axis=axis)[:, None]
    return arr


def normalize1d(arr):
    """ normalize array arr """
    arr -= np.min(arr)
    arr /= np.max(arr)
    return arr


def interpolate_intensity(mu, i):
    """ interpolate the stellar intensity for given limb distance mu """
    return interp1d(i.keys().values, i.values, kind=rw.config['interpolation_method'], fill_value=0, bounds_error=False, copy=False)(mu).swapaxes(0, 1)


def calc_mu(phase):
    """ calculate the distance from the center of the planet to the center of the star as seen from earth """
    return par['sma'] / par['r_star'] * \
        np.sqrt(np.cos(par['inc'])**2 +
                np.sin(par['inc'])**2 * np.sin(phase)**2)


def calc_intensity(phase, intensity, min_radius, max_radius, n_radii, n_angle, spacing='equidistant'):
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
    # Step 2: Calculate mu_x and mu_y
    mu_x = par['sma'] / par['r_star'] * np.sin(par['inc']) * np.sin(phase)
    mu_x = mu_x[:, None, None] + \
        (radii[:, None] * np.cos(angles)[None, :])[None, :, :]
    mu_y = par['sma'] / par['r_star'] * \
        np.cos(par['inc']) + radii[:, None] * np.sin(angles)[None, :]

    mu = np.sqrt(mu_x**2 + mu_y[None, :, :]**2)
    # Step 3: Average specific intensity, outer points weight more, as the area is larger
    intens = interpolate_intensity(mu, intensity)
    intens = np.average(intens, axis=3)
    intens = np.average(intens, axis=2, weights=radii)
    return intens


def maximum_phase():
    """ The maximum phase for which the planet is still completely inside the stellar disk """
    # This is the inverse of calc_mu(maximum_phase()) = 1.0
    return np.arcsin(np.sqrt(((par['r_star'] - par['r_planet'] - par['h_atm']) / (
        par['sma'] * np.sin(par['inc'])))**2 - np.tan(par['inc'])**-2))


def specific_intensities(phase, intensity, n_radii=11, n_angle=7):
    """
    Calculate the specific intensities of the star covered by planet and atmosphere, and only atmosphere respectively,
    over the different phases of transit
    phase: phases (in radians) of the transit, with 0 at transit center
    intensity: specific intensity profile of the host star, a pandas DataFrame with keys from 0.0 to 1.0
    n_radii: number of radii to sample, if tuple use n_radii[0] for i_planet and n_radii[1] for i_atm
    n_angle: number of angles to sample, if tuple use n_angle[0] for i_planet and n_angle[1] for i_atm
    """
    # Allow user to specify different n_radii and n_angle for i_planet and i_atm
    if isinstance(n_radii, (float, int)):
        n_radii = (n_radii, n_radii)
    if isinstance(n_angle, (float, int)):
        n_angle = (n_angle, n_angle)

    i_planet = calc_intensity(
        phase, intensity, 0, (par['r_planet'] + par['h_atm']) / par['r_star'], n_radii[0], n_angle[0])
    i_atm = calc_intensity(
        phase, intensity, par['r_planet'] / par['r_star'], (par['r_planet'] + par['h_atm']) / par['r_star'], n_radii[1], n_angle[1])
    # Alternative version that only uses the center of the planet
    # Faster but less precise (significantly?)
    #mu = calc_mu(phase)
    #intensity = interpolate_intensity(mu, intensity)
    return i_planet, i_atm


def generate_spectrum(wl, telluric, flux, intensity):
    """ Generate a fake spectrum """
    # Load planet spectrum
    planet = rw.load_input(wl)
    planet = gaussbroad(planet, sigma)
    x = np.arange(len(wl), dtype=np.float32)
    rand = np.random.rand(3, n_phase).astype(np.float32)

    # Amplitude
    amplitude = rand[0] / snr
    # Period in pixels
    period = 500 + 1500 * rand[1]
    # Phase in radians
    phase_max = maximum_phase()
    phase = np.linspace(-phase_max, phase_max, n_phase)
    # Specific intensities
    i_planet, i_atm = specific_intensities(phase, intensity)
    # Generate correlated noise
    error = np.cos(x[None, :] / period[:, None] * 2 * np.pi +
                   phase[:, None]) * amplitude[:, None]
    # Observed spectrum
    obs = (flux[None, :] - i_planet * sigma_p + i_atm *
           sigma_a * planet[None, :]) * telluric * (1 + error)
    # Generate noise
    noise = np.random.randn(n_phase, len(wl)) / snr

    obs = -gaussbroad(obs, sigma) * (1 + noise)
    """
    #Use stellar spectrum to continuum normalize
    norm = star_int[0.0] / flux
    obs = obs / norm[None, :]
    obs = obs / np.max(obs, axis=1)[:, None]
    """
    return obs


if __name__ == '__main__':
    # Step 1: Read data
    print("Loading data")
    rw = read_write(dtype=np.float32)
    par = rw.load_parameters()

    # Relative area of the stelar disk covered by the planet and atmosphere
    sigma_p = par['A_planet+atm']
    # Relative area of the atmosphere of the planet projected into the star
    sigma_a = par['A_atm']

    snr = par['snr']                       # Signal to Noise Ratio
    fwhm = par['fwhm']              # Instrumental FWHM in pixels
    sigma = 1 / 2.355 * fwhm        # Sigma of Gaussian

    # Load wavelength scale and observation
    obs, wl = rw.load_observation('all')
    n_phase = obs.shape[0]

    # Load stellar model
    #flux, star_int = rw.load_marcs(wl)
    flux, star_int = rw.load_star_model(
        wl, fwhm, 0, apply_normal=False, apply_broadening=False)

    nmu = len(star_int.keys()) - 1
    imu = np.around(np.linspace(0.1, nmu * 0.1, n_phase), decimals=1)
    # Load Tellurics
    wl_tell, tell = rw.load_tellurics(wl, n_phase // 2 + 1, apply_interp=False)
    tell = tell[n_phase // 2]  # central tellurics, has no rv shift ?

    print("Calculating intermediary data")
    # Extract orbital phase
    # TODO extract phase from observation
    phase_max = maximum_phase()
    phase = np.linspace(-phase_max, phase_max, n_phase)

    # Doppler shift telluric spectrum
    # Doppler shift
    iy = intermediary(rw.config)
    velocity = iy.rv_star(par) + iy.rv_planet(par, phase)
    tell = iy.doppler_shift(tell, wl_tell, velocity)

    # Specific intensities
    i_planet, i_atm = specific_intensities(phase, star_int)

    # Use only fake observation, for now
    # Generate fake spectrum
    obs_fake = generate_spectrum(wl, tell, flux, star_int)
    obs = obs_fake

    # Broaden everything
    tell = gaussbroad(tell, sigma)
    i_atm = gaussbroad(sigma_a * i_atm, sigma)
    i_planet = gaussbroad(sigma_p * i_planet, sigma)
    flux = gaussbroad(flux[None, :], sigma)  # Add an extra dimension

    #tell = normalize2d(tell)
    #flux = normalize2d(flux)

    f = tell
    g = -obs / i_atm - flux / i_atm * tell + i_planet / i_atm * tell

    print("Calculating solution")
    # Find best lambda
    # Step 1: make a test run without smoothing
    """
    lamb = np.zeros(len(wl), dtype=np.float32)
    lamb[:40000] = 1e3  # almost no smoothing
    lamb[40000:100000] = 1e6
    lamb[100000:130000] = 1e3
    lamb[130000:] = 1e6
    """
    lamb = 1
    sol2 = solution().solve(wl, f, g, lamb)
    
    # Step 2: find noise levels at each wavelength, aka required smoothing
    width = 500
    diff = np.zeros(len(wl))
    diff[1:] = np.exp(np.abs(np.diff(sol2)))
    diff[0] = diff[1]
    diff = maximum_filter1d(diff, width) #Use largest value in sorounding
    diff = np.clip(diff, 0, 20, out=diff) #Stop it from going infinite
    diff = np.exp(diff)
    sigma = width / 2.355 /2
    diff = gaussbroad(diff, sigma)

    #lamb = np.zeros(len(wl), dtype=np.float32)
    lamb = diff * 1e3/2

    # Step 3: Calculate solution again, this time with smoothing
    sol2 = solution().solve(wl, f, g, lamb)
    sol2 = savgol_filter(sol2, 501, 2)


    planet = rw.load_input(wl)
    # Plot
    #plt.plot(ww, rebin(sol, (nn,)), label='Best fit')
    plt.plot(wl, planet, 'r', label='Input Spectrum')
    plt.plot(wl, sol2, label='Solution')
    plt.title('Lambda = %s, S//N = %s' % (lamb, snr))
    plt.legend(loc='best')

    # save plot
    output_file = os.path.join(rw.output_dir, rw.config['file_spectrum'])
    plt.savefig(output_file, bbox_inches='tight')
    # save data
    output_file = os.path.join(rw.output_dir, rw.config['file_data_out'])
    np.savetxt(output_file, sol2)

    plt.show()
    pass
