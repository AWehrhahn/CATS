"""
Use an inverse method approach to determine the planetary transit spectrum
author: Ansgar Wehrhahn (ansgar.wehrhahn@physics.uu.se)
Based on work by Nikolai Piskunov (Uppsala University)
and              Erik Aaronson (Uppsala University)
"""
import argparse
import os.path
import sys

import matplotlib.pyplot as plt
import numpy as np

import config
import intermediary as iy
import solution as sol
from stellar_db import stellar_db
from awlib.util import normalize as normalize1d
from dataset import dataset
from harps import harps
from limb_darkening import limb_darkening
from marcs import marcs
from psg import psg
from idl import idl
from synthetic import synthetic
from REDUCE import reduce

from scipy.constants import c


def prepare(target, phase):
    """ Load data from PSG if necessary """
    conf = config.load_config(target)
    psg.load_psg(conf, phase, star=False, obs=False, tell=False, planet=True)
    return np.deg2rad(phase)


def assign_module(key):
    """assign modules according to the given key string

    Some special cases use strings as identifier and are returned as such

    Parameters:
    ----------
    key : str
        key name of the module
    Raises
    ------
    AttributeError
        key not found in valid list

    Returns
    -------
    module : {data_module, str}
        The data_module represented by that key
        or a string if special actions need to be used later
    """
    modules = {'marcs': marcs, 'psg': psg, 'harps': harps, 'limb_darkening': limb_darkening,
               'combined': 'combined', 'syn': synthetic, 'ones': 'ones', 'idl': idl, 'reduce': reduce}

    if key in modules.keys():
        return modules[key]
    else:
        raise AttributeError('Module %s not found' % key)


def get_data(conf, star, planet, **kwargs):
    """ load data from various sources

    [description]

    Parameters:
    ----------
    conf : {dict}
        configuration settings
    star : {str}
        name of the star, that is observed here
    planet : {str}
        name of the planet, usually just a letter, e.g. 'b'
    **kwargs
        other settings that will be passed on to the modules
    Returns
    -------
    par, stellar, intensities, tell, obs, phase
        stellar parameter dictionary, stellar flux, specific intensities, telluric transmission, observations, and orbital phase of the planet 
    """

    # Check settings

    parameters = conf['parameters']
    star_intensities = conf['star_intensities']

    stellar = assign_module(conf['flux'])
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
    stellar = stellar.load_stellar_flux(conf, par)

    print('   - Specific intensities')
    if intensities == 'combined':
        intensities = marcs.load_limb_darkening(conf, par)
        intensities.wl = stellar.wl
        intensities.flux = intensities.flux.apply(lambda s: s * stellar)
    else:
        intensities = intensities.load_specific_intensities(
            conf, par, stellar)

    print('   - Tellurics')
    if tellurics == 'ones':
        print('      - None')
        tell = dataset(stellar.wl, np.ones_like(stellar.flux))
    else:
        tell = tellurics.load_tellurics(conf, par)

    print('   - Observations')
    obs = observation.load_observations(
        conf, par, tell, stellar, intensities, source='psg')
    phase = obs.phase

    plt.plot(phase, 'o')
    mp = iy.maximum_phase(par)
    plt.plot(np.full(len(phase), np.pi+mp), '--r')
    plt.plot(np.full(len(phase), np.pi-mp), '--r')
    plt.show()

    # Unify wavelength grid
    #TODO bad pixel determination isn't great
    #bpmap = iy.create_bad_pixel_map(obs, threshold=1e-6)
    bpmap = np.full(obs.wl.shape, False, dtype=bool)
    bpmap[obs.wl < 5600] = True
    bpmap[obs.wl > 6750] = True
    obs.wl = obs.wl[~bpmap]

    stellar.wl = intensities.wl = tell.wl = obs.wl

    return par, stellar, intensities, tell, obs, phase


def calculate(conf, par, obs, tell, flux, star_int, phase, lamb='auto'):
    """ Calculate the planetary spectrum

    Combine all data products and reduce the problem to the linear equation f*x-g = 0, where x is the planetary spectrum
    f = tellurics * specific_intensities_planetatmosphere
    g = observation - (stellar_flux - specific_intensities_planetbody) * tellurics

    Parameters:
    ----------
    conf : {dict}
        configration settings
    par : {dict}
        stellar and orbital parameters
    obs : {dataset}
        transit observations for the planet
    tell : {dataset}
        telluric transmission
    flux : {dataset}
        stellar flux
    star_int : {dataset}
        specific intensities
    phase : {np.ndarray}
        orbital phases of the planet
    Returns
    -------
    planet: np.ndarray
        Planetary transmission spectrum
    """

    """ calculate solution from input """
    print('   - Stellar specific intensities covered by planet and atmosphere')
    i_planet, i_atm = iy.specific_intensities(par, phase, star_int)

    print('   - Broaden spectra')
    # Sigma of Instrumental FWHM in pixels
    sigma = 1 / 2.355 * conf['fwhm']

    #def gaussbroad(x, y): return x
    tell.gaussbroad(sigma)
    i_atm.scale *= par['A_atm']
    i_atm.gaussbroad(sigma)
    i_planet.scale *= par['A_planet+atm']
    i_planet.gaussbroad(sigma)
    flux.gaussbroad(sigma)


    #TODO make sure everything is in barycentric or stellar rest frame
    #shift everything into the rest frame of the planet, it should be barycentric before that
    vel = -iy.rv_planet(par, obs.phase)

    tell.doppler_shift(vel)
    i_atm.doppler_shift(vel)
    i_planet.doppler_shift(vel)
    flux.doppler_shift(vel)
    obs.doppler_shift(vel)

    tell.wl = i_atm.wl = i_planet.wl = flux.wl = obs.wl

    print('   - Intermediary products f and g')
    f = tell * i_atm
    g = obs - (flux - i_planet) * tell

    if lamb == 'auto' or lamb is None:
        print('   - Finding optimal regularization parameter lambda')
        # TODO currently doesn't work as intended
        lamb = sol.best_lambda_dirty(obs.wl, f, g, lamb0=1)
        #lamb = sol.best_lambda(wl, f, g)
    print('      - Lambda: ', lamb)
    conf['lamb'] = lamb
    print('   - Solving inverse problem')
    return sol.Tikhonov(f.flux, g.flux, lamb)


def plot(conf, par,  obs, tell, flux, sol_t, source='psg'):
    """ Plot the available data products

    Plot everything

    Parameters:
    ----------
    conf : {dict}
        configuration settings
    par : {dict}
        stellar and orbital parameters
    obs : {dataset}
        transit observations
    tell : {dataset}
        telluric transmission
    flux : {dataset}
        stellar flux
    sol_t : {np.ndarray}
        planetary transmission solution
    source : {str, 'psg'}
        string identifying the source for a comparison planet spectrum
    """

    try:
        if source in ['psg']:
            planet = psg.load_input(conf, par)
            planet.wl = obs.wl
            is_planet = True
    except FileNotFoundError:
        is_planet = False

    plt.plot(tell.wl, normalize1d(tell.flux[0]), label='Telluric')
    plt.plot(obs.wl, obs.flux[-1], label='Observation')
    plt.plot(flux.wl, flux.flux[0], label='Flux')
    if is_planet:
        plt.plot(planet.wl, planet.flux, label='Planet')
    sol_t = normalize1d(sol_t)  # TODO
    plt.plot(obs.wl, sol_t, label='Solution')

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
    """ Main entry point for the ExoSpectrum programm

    Parameters:
    ----------
    star : {str}
        name of the observed star
    planet : {str}
        name of the observed planet, just the letter, e.g. 'b'
    **kwargs
        various parameters to be passed on to the modules
    """
    # Step 0: Configuration
    combo = star + planet if star is not None and planet is not None else None
    conf = config.load_config(combo, 'config.yaml')

    # in case it wasn't clear
    star = conf['name_target']
    planet = conf['name_planet']

    # Step 1: Get Data
    print('Load data')
    prepare(star + planet, 0)
    par, flux, intensities, tell, obs, phase = get_data(
        conf, star, planet, **kwargs)

    # Step 2: Calc Solution
    print('Calculate solution')
    sol_t = calculate(conf, par, obs, tell, flux,
                      intensities, phase, lamb=lamb)

    # Step 3: Output
    print('Plot')
    # TODO in a perfect world this offset wouldn't be necessary, so can we get rid of it?
    offset = 1 - max(sol_t)
    plot(conf, par, obs, tell, flux, sol_t + offset)


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
        lamb = 1000

    # TODO size of the atmosphere in units of planetar radii (scales and shifts the solution)
    atm_factor = 0.1
    try:
        main(star, planet, lamb=lamb, atm_factor=atm_factor)
    except FileNotFoundError as fnfe:
        print("Some files seem to be missing, can't complete calculation")
        print(fnfe)
