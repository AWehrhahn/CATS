"""
Use an inverse method approach to determine the planetary transit spectrum
author: Ansgar Wehrhahn (ansgar.wehrhahn@physics.uu.se)
Based on work by Nikolai Piskunov (Uppsala University)
"""

import os.path
import numpy as np
import pandas as pd

from numpy.linalg import norm
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d as gaussbroad
import matplotlib.pyplot as plt

import intermediary as iy
import solution as sol

import config


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


def generate_spectrum(conf, par, wl, telluric, flux, intensity, phase, source='psg'):
    """ Generate a fake spectrum """
    # Sigma of Instrumental FWHM in pixels
    sigma = 1 / 2.355 * par['fwhm']

    # Load planet spectrum
    if source in ['psg']:
        import psg
        planet = psg.load_input(conf, wl)
    planet = gaussbroad(planet, sigma)

    # Specific intensities
    i_planet, i_atm = iy.specific_intensities(par, phase, intensity)

    # Observed spectrum
    obs = (flux[None, :] - i_planet * par['A_planet+atm'] +
           par['A_atm'] * i_atm * planet[None, :]) * telluric
    # Generate noise
    noise = np.random.randn(len(phase), len(wl)) / par['snr'] * 0

    obs = gaussbroad(obs, sigma) * (1 + noise)
    return obs


def calculate(conf, par, wl, obs, tell, flux, star_int, phase, lamb='auto'):
    """ calculate solution from input """
    print('   - Stellar specific intensities covered by planet and atmosphere')
    i_planet, i_atm = iy.specific_intensities(par, phase, star_int)

    print('   - Broaden spectra')
    # Sigma of Instrumental FWHM in pixels
    sigma = 1 / 2.355 * conf['fwhm']

    #def gaussbroad(x, y): return x
    tell = gaussbroad(tell, sigma)
    i_atm = gaussbroad(par['A_atm'] * i_atm, sigma)
    i_planet = gaussbroad(par['A_planet+atm'] * i_planet, sigma)
    flux = gaussbroad(flux[None, :], sigma)  # Add an extra dimension

    print('   - Intermediary products f and g')
    f = tell * i_atm
    g = obs - (flux - i_planet) * tell

    if lamb == 'auto' or lamb is None:
        print('   - Finding optimal regularization parameter lambda')
        lamb = sol.best_lambda(wl, f, g)
    print('      - Lambda: ', lamb)
    print('   - Solving inverse problem')
    # return normalize1d(sol.Tikhonov(wl, f, g, lamb))
    return sol.Tikhonov(wl, f, g, lamb)


def plot(conf, par, wl, obs, fake, tell, flux, sol_t, sol_f, source='psg'):
    """ plot resulting data """
    if source in ['psg']:
        import psg
        planet = psg.load_input(conf, wl)

    plt.plot(wl, tell, label='Telluric')
    plt.plot(wl, obs[0], label='Observation')
    plt.plot(wl, flux, label='Flux')
    #plt.plot(wl, fake[0], label='Fake')
    plt.plot(wl, planet, label='Planet')
    plt.plot(wl, sol_t, label='Solution')

    plt.title('%s\nLambda = %.3f, S/N = %s' %
              (par['name_star'] + ' ' + par['name_planet'], np.mean(1e-5), conf['snr']))
    plt.xlabel('Wavelength [Å]')
    plt.ylabel('Intensity [norm.]')
    plt.legend(loc='best')

    # save plot
    output_file = os.path.join(conf['output_dir'], conf['file_spectrum'])
    if not os.path.exists(conf['output_dir']):
        os.mkdir(conf['output_dir'])
    plt.savefig(output_file, bbox_inches='tight')
    # save data
    output_file = os.path.join(conf['output_dir'], conf['file_data_out'])
    np.savetxt(output_file, sol_t)

    plt.show()


def prepare(target, phase):
    # Load data from PSG if necessary
    import psg
    conf = config.load_config(target)
    psg.load_psg(conf, phase)
    return np.deg2rad(phase)


def get_data(conf, star, planet):
    """
    Load data from specified sources
    """
    # Check settings
    parameters = conf['parameters']
    flux = conf['flux']
    intensities = conf['intensities']
    observation = conf['observation']
    tellurics = conf['tellurics']

    # Parameters
    if parameters in ['stellar_db', 'sdb']:
        import stellar_db
        par = stellar_db.load_parameters(star, planet)

    if observation in ['psg']:
        import psg
        wl_obs, obs, phase = psg.load_observation(conf)

    # Stellar Flux
    if flux in ['marcs', 'm']:
        import marcs
        wl_flux, flux = marcs.load_flux(conf)
    elif flux in ['psg']:
        import psg
        wl_flux, flux = psg.load_flux(conf)

    # Specific intensities
    if intensities in ['limb_darkening']:
        import limb_darkening
        wl_si, intensities = limb_darkening.load_intensities(
            conf, par, wl_flux, flux)

    # Tellurics
    if tellurics in ['psg']:
        import psg
        wl_tell, tell = psg.load_tellurics(conf)
    elif tellurics in ['one', 'ones'] or tellurics is None:
        wl_tell = wl_obs[0]
        tell = np.ones_like(wl_tell)

    # Unify wavelength grid
    bpmap = iy.create_bad_pixel_map(obs, threshold=1e-6)
    wl = wl_obs[0, ~bpmap]
    obs = obs[:, ~bpmap]

    flux = np.interp(wl, wl_flux, flux)
    data = np.array([np.interp(wl, wl_si, intensities[i])
                     for i in intensities.keys()]).swapaxes(0, 1)
    intensities = pd.DataFrame(data=data, columns=intensities.keys())
    tell = np.interp(wl, wl_tell, tell)

    return par, wl, flux, intensities, tell, obs, phase


def main(star, planet, lamb='auto', use_fake=False):
    """
    Main entry point for the ExoSpectrum Programm
    """
    # Configuration
    conf = config.load_config(star + planet, 'config.yaml')
    # Step 1: Get Data
    par, wl, flux, intensities, tell, obs, phase = get_data(conf, star, planet)
    # Step 2: Calc Solution
    sol_t = calculate(conf, par, wl, obs, tell, flux,
                      intensities, phase, lamb=lamb)

    if use_fake:
        fake = generate_spectrum(conf, par, wl, tell, flux, intensities, phase)
        sol_f = calculate(conf, par, wl, fake, tell, flux,
                          intensities, phase, lamb=lamb)
    else:
        fake = sol_f = None

    # Step 3: Output
    plot(conf, par, wl, obs, fake, tell, flux, sol_t, sol_f)
    pass


if __name__ == '__main__':
    main('Trappist-1', 'c', lamb=1e-4)
