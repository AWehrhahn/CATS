"""
Use an inverse method approach to determine the planetary transit spectrum
author: Ansgar Wehrhahn (ansgar.wehrhahn@physics.uu.se)
Based on work by Nikolai Piskunov (Uppsala University)
"""
import argparse
import os.path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d as gaussbroad

import config
import intermediary as iy
import solution as sol
import stellar_db
from awlib.util import normalize as normalize1d
from awlib.util import interpolate_DataFrame
from harps import harps
from limb_darkening import limb_darkening
from marcs import marcs
from psg import psg
from synthetic import synthetic


def prepare(target, phase):
    """ Load data from PSG if necessary """
    conf = config.load_config(target)
    psg.load_psg(conf, phase)
    return np.deg2rad(phase)


def assign_module(key):
    """ assign module according to given key """
    #some special cases still use strings as identifiers
    modules = {'marcs': marcs, 'psg': psg, 'harps': harps, 'limb_darkening': limb_darkening,
               'combined': 'combined', 'syn': synthetic, 'ones': 'ones'}

    if key in modules.keys():
        return modules[key]
    else:
        raise AttributeError('Module %s not found' % key)


def get_data(conf, star, planet, **kwargs):
    """
    Load data from specified sources
    """
    # Check settings

    parameters = conf['parameters']
    star_intensities = conf['star_intensities']

    flux = assign_module(conf['flux'])
    intensities = assign_module(conf['intensities'])
    observation = assign_module(conf['observation'])
    tellurics = assign_module(conf['tellurics'])

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
    wl_flux, flux = flux.load_stellar_flux(conf, par)

    print('   - Specific intensities')
    if intensities == 'combined':
        wl_f, factors = marcs.load_limb_darkening(conf, par)
        factors = interpolate_DataFrame(wl_flux, wl_f, factors)
        intensities = factors.apply(lambda s: s * flux)
        wl_si = wl_flux
    else:
        wl_si, intensities = intensities.load_specific_intensities(
            conf, par, wl_flux, flux)

    print('   - Tellurics')
    if tellurics == 'ones':
        wl_tell = wl_flux
        tell = np.ones_like(wl_tell)
    else:
        wl_tell, tell = tellurics.load_tellurics(conf, par)

    print('   - Observations')
    wl_obs, obs, phase = observation.load_observations(
        conf, par, wl_tell, tell, wl_flux, flux, intensities, source='psg')

    # Unify wavelength grid
    bpmap = iy.create_bad_pixel_map(obs, threshold=1e-6)
    wl = wl_obs[~bpmap]
    obs = obs[:, ~bpmap]

    flux = np.interp(wl, wl_flux, flux)
    intensities = interpolate_DataFrame(wl, wl_si, intensities)
    tell = np.interp(wl, wl_tell, tell)

    # TODO DEBUG
    if observation is harps:
        obs = harps.flux_calibration(conf, par, wl, obs)
    # TODO END_DEBUG

    return par, wl, flux, intensities, tell, obs, phase


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
        lamb = sol.best_lambda_dirty(wl, f, g, lamb0=1)
        #lamb = sol.best_lambda(wl, f, g)
    print('      - Lambda: ', lamb)
    conf['lamb'] = lamb
    print('   - Solving inverse problem')
    return sol.Tikhonov(wl, f, g, lamb)

def plot(conf, par, wl, obs, tell, flux, sol_t, source='psg'):
    """ plot resulting data """
    try:
        if source in ['psg']:
            planet = psg.load_input(conf, par, wl)
            is_planet = True
    except FileNotFoundError:
        is_planet = False

    plt.plot(wl, normalize1d(tell), label='Telluric')
    plt.plot(wl, normalize1d(obs[0]), label='Observation')
    plt.plot(wl, normalize1d(flux), label='Flux')
    if is_planet:
        plt.plot(wl, planet, label='Planet')
    sol_t = normalize1d(sol_t)  # TODO
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


def main(star, planet, lamb='auto', **kwargs):
    """
    Main entry point for the ExoSpectrum Programm
    """
    # Step 0: Configuration
    combo = star + planet if star is not None and planet is not None else None
    conf = config.load_config(combo, 'config.yaml')

    # in case it wasn't clear
    star = conf['name_target']
    planet = conf['name_planet']

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
        lamb = 1000000

    # TODO size of the atmosphere in units of planetar radii (scales and shifts the solution)
    atm_factor = 0.1
    try:
        main(star, planet, lamb=lamb, atm_factor=atm_factor)
    except FileNotFoundError as fnfe:
        print("Some files seem to be missing, can't complete calculation")
        print(fnfe)
