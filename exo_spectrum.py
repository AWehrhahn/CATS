"""
Use an inverse method approach to determine the planetary transit spectrum
author: Ansgar Wehrhahn (ansgar.wehrhahn@physics.uu.se)
Based on work by Nikolai Piskunov (Uppsala University)
"""
import sys
import argparse
import os.path
import numpy as np
import pandas as pd

from scipy.ndimage.filters import gaussian_filter1d as gaussbroad
import matplotlib.pyplot as plt

from awlib.util import interpolate_DataFrame

import intermediary as iy
import solution as sol

import config
import psg
import harps
import marcs
import stellar_db
import limb_darkening
import synthetic


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
        # TODO currently doesn't work as intended
        lamb = sol.best_lambda(wl, f, g)
    print('      - Lambda: ', lamb)
    conf['lamb'] = lamb
    print('   - Solving inverse problem')
    # return normalize1d(sol.Tikhonov(wl, f, g, lamb))
    return sol.Tikhonov(wl, f, g, lamb)


def plot(conf, par, wl, obs, tell, flux, sol_t, source='psg'):
    """ plot resulting data """
    try:
        if source in ['psg']:
            planet = psg.load_input(conf, wl)
            is_planet = True
    except FileNotFoundError:
        is_planet = False

    plt.plot(wl, tell, label='Telluric')
    plt.plot(wl, obs[0], label='Observation')
    plt.plot(wl, flux, label='Flux')
    if is_planet:
        plt.plot(wl, planet, label='Planet')
    # sol_t = normalize1d(sol_t)  # TODO
    plt.plot(wl, sol_t, label='Solution')

    plt.title('%s\nLambda = %.3g, S/N = %s' %
              (par['name_star'] + ' ' + par['name_planet'], conf['lamb'], conf['snr']))
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
    conf = config.load_config(target)
    psg.load_psg(conf, phase)
    return np.deg2rad(phase)


def get_data(conf, star, planet, **kwargs):
    """
    Load data from specified sources
    """
    # Check settings
    parameters = conf['parameters']
    flux = conf['flux']
    intensities = conf['intensities']
    observation = conf['observation']
    tellurics = conf['tellurics']
    star_intensities = conf['star_intensities']

    # Parameters
    print('   - Parameters')
    if parameters in ['stellar_db', 'sdb']:
        par = stellar_db.load_parameters(star, planet, **kwargs)

    if not isinstance(star_intensities, list):
        if star_intensities in ['geom']:
            imu = np.geomspace(1, 0.0001, num=20)
            imu[-1] = 0
            conf['star_intensities'] = imu
            star_intensities = imu

    print('   - Stellar flux')
    if flux in ['marcs', 'm']:
        wl_flux, flux = marcs.load_flux(conf, par)
    elif flux in ['psg']:
        wl_flux, flux = psg.load_flux(conf)
    elif flux in ['harps']:
        wl_flux, flux = harps.load_flux(conf, par)


    print('   - Specific intensities')
    if intensities in ['limb_darkening']:
        wl_si, intensities = limb_darkening.load_intensities(
            conf, par, wl_flux, flux)
    elif intensities in ['marcs']:
        wl_si, intensities = marcs.load_intensities(conf, par)
    elif intensities in ['combined']:
        wl_f, factors = marcs.load_limb_darkening(conf, par)
        factors = interpolate_DataFrame(wl_flux, wl_f, factors)
        intensities = factors.apply(lambda s: s * flux)
        wl_si = wl_flux

    print('   - Tellurics')
    if tellurics in ['psg']:
        wl_tell, tell = psg.load_tellurics(conf)
    elif tellurics in ['harps']:
        wl_tell, tell = harps.load_tellurics(conf, par)
    elif tellurics in ['one', 'ones'] or tellurics is None:
        wl_tell = wl_flux
        tell = np.ones_like(wl_tell)

    print('   - Observations')
    if observation in ['psg']:
        wl_obs, obs, phase = psg.load_observation(conf)
    elif observation in ['harps']:
        wl_obs, obs, phase = harps.load_observation(conf, par)
    elif observation in ['syn', 'synthetic', 'fake']:
        wl_obs, obs, phase = synthetic.generate_spectrum(
            conf, par, wl_tell, tell, wl_flux, flux, intensities, source='psg')


    # Unify wavelength grid
    bpmap = iy.create_bad_pixel_map(obs, threshold=1e-6)
    # TODO for PSG at least wl lower than 8000 Å are bad
    bpmap[(wl_obs <= 8100)] = True
    wl = wl_obs[~bpmap]
    obs = obs[:, ~bpmap]

    flux = np.interp(wl, wl_flux, flux)
    intensities = interpolate_DataFrame(wl, wl_si, intensities)
    tell = np.interp(wl, wl_tell, tell)

    # TODO DEBUG

    # Adding noise to the observation
    #noise = np.random.random_sample(obs.shape) / conf['snr']
    #obs *= 1 + noise

    # Scaling the stellar flux to the same size as the observations
    #factor = max(obs[0]) / max(flux)
    #flux = flux * factor
    #intensities = intensities.apply(lambda s: s * factor)

    # TODO END_DEBUG

    return par, wl, flux, intensities, tell, obs, phase


def main(star, planet, lamb='auto', **kwargs):
    """
    Main entry point for the ExoSpectrum Programm
    """
    # Step 0: Configuration
    if star is not None and planet is not None:
        combo = star + planet
    else:
        combo = None
    conf = config.load_config(combo, 'config.yaml')
    if combo is None:
        star = conf['name_target']
        planet = conf['name_planet']

    # TODO
    psg.load_psg(conf, [180, 180.04, 180.08, 180.12, 180.16, 180.20, 180.24, 180.28, 180.32, 180.36, 180.4,
                        179.96, 179.92, 179.88, 179.84, 179.80, 179.76, 179.72, 179.68, 179.64])

    # Step 1: Get Data
    print('Load data')
    par, wl, flux, intensities, tell, obs, phase = get_data(
        conf, star, planet, **kwargs)

    # Step 2: Calc Solution
    print('Calculate solution')
    sol_t = calculate(conf, par, wl, obs, tell, flux,
                      intensities, phase, lamb=lamb)

    # Step 3: Output
    print('Plot')
    # TODO in a perfect world this offset wouldn't be necessary, so can we get rid of it?
    offset = 1 - max(sol_t)
    plot(conf, par, wl, obs, tell, flux, sol_t + offset)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(
            description='Extract the planetary transmittance spectrum, from one or more transit observations.')
        parser.add_argument('star', type=str, help='The observed star')
        parser.add_argument(
            'planet', type=str, help='The letter of the planet (default=b)', nargs='?', default='b')
        parser.add_argument('-l', '--lambda', type=str,
                            help='Regularization parameter lambda (default=auto)', default='auto', dest='lamb')

        args = parser.parse_args()
        star = args.star
        planet = args.planet
        lamb = args.lamb
        if lamb != 'auto':
            try:
                lamb = float(lamb)
            except ValueError:
                print('Invalid value for -l/-lambda')
                exit()
        else:
            # TODO
            print('WARNING: lambda=auto does currently not work properly')
    else:
        star = None
        planet = None
        #lamb = 'auto'
        lamb = 0.1

    # TODO size of the atmosphere in units of planetar radii (scales and shifts the solution)
    atm_factor = 0.1
    try:
        main(star, planet, lamb=lamb, atm_factor=atm_factor)
    except FileNotFoundError as fnfe:
        print("Some files seem to be missing, can't complete calculation")
        print(fnfe)
