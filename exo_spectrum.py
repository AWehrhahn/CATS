"""
Use an inverse method approach to determine the planetary transit spectrum
author: Ansgar Wehrhahn (ansgar.wehrhahn@physics.uu.se)
Based on work by Nikolai Piskunov (Uppsala University)
"""

import os.path
import subprocess
import numpy as np

from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d as gaussbroad
from scipy.optimize import fsolve, minimize
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve, inv
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
    bpmap = np.all(obs == 0, axis=0) | np.all(
        obs == 1, axis=0)  # Bad Pixel Map
    # remove them
    wl = wl[~bpmap]
    obs = obs[:, ~bpmap]

    # Load tellurics
    print('   - Tellurics')
    if not os.path.exists(os.path.join(rw.intermediary_dir, rw.config['file_telluric'] + '_fit.fits')) or rw.renew_all:
        print('      Fit tellurics with molecfit')
        rw.convert_keck_fits()
        iy.fit_tellurics(verbose=True)
    else:
        print('      Use existing telluric fit')
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
    # Find best lambda
    sol = solution(dtype=np.float32)
    # TODO figure out if there is a better way
    # maximum value of the solution should be 1, then there are no spikes
    lamb = 1e2
    #mesg = 'Use fixed lambda'

    # We are using generalized Tikhonov regularization, with regularization matrix = differntial operator

    # Using Generalized Cross Validation
    n = len(wl)
    #I = np.matlib.identity(n)
    a = c = np.full(n - 1, -1)
    b = np.full(n, 2)
    b[0] = b[-1] = 1
    L = diags([a, b, c], offsets=[-1, 0, 1])
    A = diags(f[0], 0)  # + lamb*L
    b = g[0]
    I = diags(np.ones(n), 0)

    #sol_tx = spsolve(A, b)

    def xl(l): 
        return spsolve(A + l * L, b)

    def t_inv(t): 
        return np.where(t != 0, 1 / t / len(t[t != 0]), 0)

    def influence(l): 
        return (A + l * L) * np.dot(xl(l), t_inv(b))

    def gcv(l): 
        if isinstance(l, (np.ndarray, list)):
            l = l[0]
        return np.linalg.norm(A * xl(l) - b) / np.sum(1 - influence(l).diagonal())

    sol_t = minimize(gcv, lamb, tol=1e-16, options={'disp':True})
    sol_tx = sol_t.x
    pass
    """
    # Using dirty limitation of max(solution) == 1
    def func(x): return sol.solve(wl, f, g, np.abs(x)).max() - 1
    lamb, info, ier, mesg = fsolve(func, x0=lamb, full_output=True)
    lamb = np.abs(lamb[0])
    """

    sol2 = sol.solve(wl, f, g, lamb)
    sol2 = np.clip(sol2, 0, 1)
    #print('   -', mesg)
    print('   - Best fit lambda: ', lamb)

    """
    # Step 2: find noise levels at each wavelength, aka required smoothing
    # TODO find some good consistent way to do this
    width = 100
    sigma = width / 2.355
    low = 1
    top = 1e4

    diff = np.zeros(len(wl))
    diff[1:] = np.exp(np.abs(np.diff(sol2)))
    diff[0] = diff[1]
    diff = gaussbroad(diff, width)
    diff = diff**(np.log(top/low)/np.log(np.max(diff))) * low

    lamb = diff

    # Step 3: Calculate solution again, this time with smoothing
    sol2 = sol.solve(wl, f, g, lamb)
    """

    planet = rw.load_input(wl * 0.25)
    # Plot
    plt.plot(wl, tell[0], label='Telluric')
    plt.plot(wl, planet, 'r', label='Planet')
    plt.plot(wl, sol2, label='Solution')
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
    np.savetxt(output_file, sol2)

    plt.show()
    pass
