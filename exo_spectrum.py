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
    planet = rw.load_input(wl*0.25)
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
    lamb = 3.14
    #mesg = 'Use fixed lambda'

    def eta(sol2):
        return np.linalg.norm(sol2)**2

    def rho(sol2):
        return np.linalg.norm(f*sol2[None, :]-g)**2

    def eta_prime(lamb, sol2):
        a1 = np.array([f[0], np.full(len(wl), lamb, dtype=np.float32)], dtype=np.float32)
        a2 = np.array([(f*sol2[None, :] - g)[0], np.zeros(len(wl), dtype=np.float32)], dtype=np.float32)
        z = minimize(lambda z: np.linalg.norm(a1*z[None, :] - a2), x0 = np.ones(len(wl), dtype=np.float32), tol=1e-6)
        z = z.x
        eta = 4/lamb * sol2 * z
        return eta

    def curv(lamb):
        sol2 = sol.solve(wl, f, g, lamb)
        e = eta(sol2)
        r = rho(sol2)
        p = eta_prime(lamb, sol2)
        
        return 2*e*r/p * (lamb**2 *p*r + 2*lamb*e*r+lamb**4*e*p)/(lamb**2*e**2+r**2)**1.5
    
    print(curv(lamb))
    """
    # Using Generalized Cross Validation
    def gcv(lamb): 
        return np.linalg.norm(f * sol.solve(wl, f, g, np.abs(lamb)) - g) / (len(wl) - np.abs(lamb))
    lamb = minimize(gcv, x0=lamb, method='Powell', options={'disp': True})
    mesg = lamb.message
    lamb = np.abs(lamb.x[0])
    """

    """
    # Using dirty limitation of max(solution) == 1
    def func(x): return sol.solve(wl, f, g, np.abs(x)).max() - 1
    lamb, info, ier, mesg = fsolve(func, x0=lamb, full_output=True)
    lamb = np.abs(lamb[0])
    """
    sol2 = sol.solve(wl, f, g, lamb)
    sol2 = np.clip(sol2, 0, 1)
    print('   -', mesg)
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

    planet = rw.load_input(wl*0.25)
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
