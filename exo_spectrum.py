"""
Use an inverse method approach to determine the planetary transit spectrum
author: Ansgar Wehrhahn (ansgar.wehrhahn@physics.uu.se)
Based on work by Nikolai Piskunov (Uppsala University)
"""

import os.path
import subprocess
import numpy as np

from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d as gaussbroad, maximum_filter1d
from scipy.optimize import fsolve
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

def generate_spectrum(wl, telluric, flux, intensity):
    """ Generate a fake spectrum """
    # Load planet spectrum
    planet = rw.load_input(wl*0.25)
    planet = gaussbroad(planet, sigma)
    #x = np.arange(len(wl), dtype=np.float32)
    #rand = np.random.rand(3, n_phase).astype(np.float32)

    # Amplitude
    #amplitude = rand[0] / snr
    # Period in pixels
    #period = 500 + 1500 * rand[1]
    # Phase in radians
    phase_max = iy.maximum_phase()
    phase = np.linspace(-phase_max, phase_max, n_phase)
    # Specific intensities
    i_planet, i_atm = iy.specific_intensities(phase, intensity)
    # Generate correlated noise
    #error = np.cos(x[None, :] / period[:, None] * 2 * np.pi +
    #               phase[:, None]) * amplitude[:, None]
    # Observed spectrum
    obs = (flux[None, :] - i_planet * sigma_p + i_atm *
           sigma_a * planet[None, :]) * telluric
    # Generate noise
    noise = np.random.randn(n_phase, len(wl)) / snr

    obs = gaussbroad(obs, sigma) * (1 + noise)
    return obs

if __name__ == '__main__':
    # Step 1: Read data
    print("Loading data")
    rw = read_write(dtype=np.float32)
    par = rw.load_parameters()
    iy = intermediary(rw.config, par, dtype=np.float32)

    # Relative area of the stelar disk covered by the planet and atmosphere
    sigma_p = par['A_planet+atm']
    # Relative area of the atmosphere of the planet projected into the star
    sigma_a = par['A_atm']

    snr = par['snr']                       # Signal to Noise Ratio
    fwhm = par['fwhm']              # Instrumental FWHM in pixels
    sigma = 1 / 2.355 * fwhm        # Sigma of Gaussian
    n_phase = par['n_exposures']

    # Load wavelength scale and observation
    wl, obs = rw.load_observation('all')
    if obs.ndim == 1:
        obs = obs[None, :]

    # Load tellurics
    if not os.path.exists(os.path.join(rw.intermediary_dir, rw.config['file_telluric'] + '_fit.fits')) or rw.renew_all:
        print('Fit tellurics with molecfit')
        rw.convert_keck_fits()
        iy.fit_tellurics(verbose=True)
    else:
        print('Use existing telluric fit')
    wl_tell, tell = rw.load_tellurics()

    # Load stellar model
    flux, star_int = rw.load_star_model(
        wl*0.25, fwhm, 0, apply_normal=True, apply_broadening=False)

    print("Calculating intermediary data")
    # Extract orbital phase
    # TODO extract phase from observation
    phase_max = iy.maximum_phase()
    phase = np.linspace(-phase_max, phase_max, n_phase)

    # Doppler shift telluric spectrum
    # Doppler shift
    velocity = iy.rv_star() + iy.rv_planet(phase)
    tell = iy.doppler_shift(tell, wl_tell, velocity)
    tell = interp1d(wl_tell, tell, fill_value='extrapolate')(wl)

    # Specific intensities
    i_planet, i_atm = iy.specific_intensities(phase, star_int)

    # Use only fake observation, for now
    # Generate fake spectrum
    obs = generate_spectrum(wl, tell, flux, star_int)


    # Broaden everything
    tell = gaussbroad(tell, sigma)
    i_atm = gaussbroad(sigma_a * i_atm, sigma)
    i_planet = gaussbroad(sigma_p * i_planet, sigma)
    flux = gaussbroad(flux[None, :], sigma)  # Add an extra dimension

    f = tell * i_atm
    g = obs - flux * tell + i_planet * tell

    print("Calculating solution")
    # Find best lambda
    sol = solution(dtype=np.float32)
    # maximum value of the solution should be 1, then there are no spikes
    lamb = 1e2
    func = lambda x: sol.solve(wl, f, g, x).max() - 1
    lamb, info, ier, mesg = fsolve(func, x0=lamb, full_output=True)
    sol2 = sol.solve(wl, f, g, lamb)

    """
    # Step 2: find noise levels at each wavelength, aka required smoothing
    # TODO find some good consistent way to do this
    width = 1000
    sigma = width / 2.355
    low = 1e4
    top = 1e7

    diff = np.zeros(len(wl))
    diff[1:] = np.exp(np.abs(np.diff(sol2)))
    diff[0] = diff[1]
    diff = gaussbroad(diff, width)
    diff = diff**(np.log(top/low)/np.log(np.max(diff))) * low

    lamb = diff

    # Step 3: Calculate solution again, this time with smoothing
    sol2 = sol.solve(wl, f, g, lamb)
    """

    planet = rw.load_input(wl*0.25)
    # Plot
    #plt.plot(ww, rebin(sol, (nn,)), label='Best fit')
    plt.plot(wl, planet, 'r', label='Planet')
    #plt.plot(wl, normalize1d(obs[0]), label='Observation')
    plt.plot(wl, tell[0], label='Telluric')
    plt.plot(wl, sol2, label='Solution')
    plt.title('Lambda = %s, S/N = %s' % (np.mean(lamb), snr))
    plt.legend(loc='best')

    # save plot
    output_file = os.path.join(rw.output_dir, rw.config['file_spectrum'])
    plt.savefig(output_file, bbox_inches='tight')
    # save data
    output_file = os.path.join(rw.output_dir, rw.config['file_data_out'])
    np.savetxt(output_file, sol2)

    plt.show()
    pass
