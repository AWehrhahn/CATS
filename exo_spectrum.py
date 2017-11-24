"""
Use an inverse method approach to determine the planetary transit spectrum
author: Ansgar Wehrhahn (ansgar.wehrhahn@physics.uu.se)
Based on work by Nikolai Piskunov (Uppsala University)
"""

import os.path
import timeit
import subprocess
import numpy as np


from numpy.linalg import norm
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d as gaussbroad
from scipy.optimize import fsolve, minimize, curve_fit
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve, inv, svds
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
    arr -= np.min(arr, axis=axis)[:, None]
    arr /= np.max(arr, axis=axis)[:, None]
    return arr


def normalize1d(arr):
    """ normalize array arr """
    arr -= np.min(arr)
    arr /= np.max(arr)
    return arr


def generate_spectrum(wl, telluric, flux, intensity, phase):
    """ Generate a fake spectrum """
    snr = par['snr']                       # Signal to Noise Ratio

    # Load planet spectrum
    planet = rw.load_input(wl * 0.25)
    planet = gaussbroad(planet, sigma)

    # Specific intensities
    i_planet, i_atm = iy.specific_intensities(phase, intensity)
    # Observed spectrum
    obs = (flux[None, :] - i_planet * par['A_planet+atm'] +
           par['A_atm'] * i_atm * planet[None, :]) * telluric
    # Generate noise
    noise = np.random.randn(len(phase), len(wl)) / snr

    obs = gaussbroad(obs, sigma) * (1 + noise)
    return obs


if __name__ == '__main__':
    # Step 1: Read data
    print("Loading data")
    rw = read_write(dtype=np.float32)
    print('   - Orbital parameters')
    par = rw.load_parameters()
    iy = intermediary(rw.config, par, dtype=np.float32)

    # Sigma of Instrumental FWHM in pixels
    sigma = 1 / 2.355 * par['fwhm']
    n_exposures = par['n_exposures']       # Number of observations per transit

    # Load wavelength scale and observation and phase information
    print('   - Observation')
    wl, obs, phase = rw.load_observation('all')
    if obs.ndim == 1:  # Ensure that obs is 2 dimensional
        obs = obs[None, :]

    # Find and remove bad pixels/areas
    print('   - Find and remove bad pixels')
    # Find all pixels that are always 0 or always 1
    bpmap = iy.create_bad_pixel_map(obs)
    # remove them
    wl = wl[~bpmap]
    obs = obs[:, ~bpmap]

    # Load tellurics
    print('   - Tellurics')
    if not os.path.exists(os.path.join(rw.intermediary_dir, rw.config['file_telluric'] + '_fit.fits')) or rw.renew_all:
        print('      - Fit tellurics with molecfit')
        rw.convert_keck_fits()
        iy.fit_tellurics(verbose=True)
    else:
        print('      - Use existing telluric fit')
    wl_tell, tell = rw.load_tellurics()

    # Load stellar model
    print('   - Stellar model')
    #flux, star_int = rw.load_star_model(wl)
    flux, star_int = rw.load_marcs(wl)

    print("Calculating intermediary data")
    # Doppler shift telluric spectrum
    print('   - Doppler shift tellurics')
    velocity = iy.rv_star() + iy.rv_planet(phase)
    tell = iy.doppler_shift(tell, wl_tell, velocity)
    tell = interp1d(wl_tell, tell, fill_value='extrapolate')(wl)

    # Specific intensities
    print('   - Stellar specific intensities covered by planet and atmosphere')
    i_planet, i_atm = iy.specific_intensities(phase, star_int)

    # Use only fake observation, for now
    # Generate fake spectrum
    print('   - Synthetic observation')
    obs = generate_spectrum(wl, tell, flux, star_int, phase)

    # Broaden everything
    tell = gaussbroad(tell, sigma)
    i_atm = gaussbroad(par['A_atm'] * i_atm, sigma)
    i_planet = gaussbroad(par['A_planet+atm'] * i_planet, sigma)
    flux = gaussbroad(flux[None, :], sigma)  # Add an extra dimension

    # Sum (f * X - g)**2 = min
    # Avoid division
    print('   - Intermediary products f and g')
    f = tell * i_atm
    g = obs - (flux - i_planet) * tell

    print("Calculating solution")
    sol = solution(dtype=np.float32)
    print('   - Finding optimal regularization parameter lambda')
    lamb = sol.best_lambda(wl, f, g)
    lamb_dirty = sol.best_lambda_dirty(wl, f, g)
    print('      - L Curve: ', lamb)
    print('      - Dirty Hack: ', lamb_dirty)

    print('   - Solving inverse problem')
    sol_t = sol.Tikhonov(wl, f, g, lamb)
    sol_td = sol.Tikhonov(wl, f, g, lamb_dirty)
    sol_f = sol.Franklin(wl, f, g, lamb)
    planet = rw.load_input(wl * 0.25)

    #Plotting
    plt.plot(wl, planet, label='Planet')
    plt.plot(wl, sol_t, label='Tikhonov')
    plt.plot(wl, sol_td, label='Dirty')
    plt.plot(wl, sol_f, label='Franklin')
    plt.legend(loc='best')

    plt.show()

    planet = rw.load_input(wl * 0.25)
    # Plot
    plt.plot(wl, tell[0], label='Telluric')
    plt.plot(wl, planet, 'r', label='Planet')
    plt.plot(wl, sol_t, label='Solution')
    plt.title('%s\nLambda = %.3f, S/N = %s' %
              (par['name_star'] + ' ' + par['name_planet'], np.mean(lamb), par['snr']))
    plt.xlabel('Wavelength [Å]')
    plt.ylabel('Intensity [norm.]')
    plt.legend(loc='best')

    # save plot
    output_file = os.path.join(rw.output_dir, rw.config['file_spectrum'])
    plt.savefig(output_file, bbox_inches='tight')
    # save data
    output_file = os.path.join(rw.output_dir, rw.config['file_data_out'])
    np.savetxt(output_file, sol_f)

    plt.show()
    pass
