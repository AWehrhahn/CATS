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

import intermediary as iy
import solution as sol

import config
import psg
import marcs
import stellar_db
import limb_darkening


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
    sigma = 1 / 2.355 * conf['fwhm']

    try:
        # Load planet spectrum
        if source in ['psg']:
            planet = psg.load_input(conf, wl)
    except FileNotFoundError:
        print('No planet spectrum for synthetic observation found')
        raise FileNotFoundError
    
    planet = gaussbroad(planet, sigma)

    # Specific intensities
    i_planet, i_atm = iy.specific_intensities(par, phase, intensity)

    # Observed spectrum
    obs = (flux[None, :] - i_planet * par['A_planet+atm'] +
           par['A_atm'] * i_atm * planet[None, :]) * telluric
    # Generate noise
    noise = np.random.randn(len(phase), len(wl)) / conf['snr']
    # TODO
    #noise = 0

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
        lamb = sol.best_lambda(wl, f, g) #TODO currently doesn't work as intended
    print('      - Lambda: ', lamb)
    conf['lamb'] = lamb
    print('   - Solving inverse problem')
    # return normalize1d(sol.Tikhonov(wl, f, g, lamb))
    return sol.Tikhonov(wl, f, g, lamb)


def plot(conf, par, wl, obs, fake, tell, flux, sol_t, sol_f, source='psg'):
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
    sol_t = normalize1d(sol_t) #TODO
    plt.plot(wl, sol_t, label='Solution')
    if fake is not None:
        plt.plot(wl, fake[0], label='Fake')
        plt.plot(wl, sol_f, label='Sol Fake')

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

    # Parameters
    print('   - Parameters')
    if parameters in ['stellar_db', 'sdb']:
        par = stellar_db.load_parameters(star, planet, **kwargs)

    print('   - Observations')
    if observation in ['psg']:
        wl_obs, obs, phase = psg.load_observation(conf)

    print('   - Stellar flux')
    if flux in ['marcs', 'm']:
        imu = np.geomspace(1, 0.0001, num=20)
        imu[-1] = 0
        conf['star_intensities'] = imu
        
        wl_flux, flux = marcs.load_flux(conf, par)
    elif flux in ['psg']:
        wl_flux, flux = psg.load_flux(conf)

    print('   - Specific intensities')
    #interpolation points
    #TODO set up config values for this

    if intensities in ['limb_darkening']:
        wl_si, intensities = limb_darkening.load_intensities(
            conf, par, wl_flux, flux)
    elif intensities in ['marcs']:
        wl_si, intensities = marcs.load_limb_darkening(conf, par)

    print('   - Tellurics')
    if tellurics in ['psg']:
        wl_tell, tell = psg.load_tellurics(conf)
    elif tellurics in ['one', 'ones'] or tellurics is None:
        wl_tell = wl_obs[0]
        tell = np.ones_like(wl_tell)

    # Unify wavelength grid
    bpmap = iy.create_bad_pixel_map(obs, threshold=1e-6)
    bpmap[(wl_obs[0] <= 8100)] = True #TODO for PSG at least wl lower than 8000 Å are bad
    wl = wl_obs[0, ~bpmap]
    obs = obs[:, ~bpmap]

    flux = np.interp(wl, wl_flux, flux)
    data = np.array([np.interp(wl, wl_si, intensities[i])
                     for i in intensities.keys()]).swapaxes(0, 1)
    intensities = pd.DataFrame(data=data, columns=intensities.keys())
    tell = np.interp(wl, wl_tell, tell)

    #TODO DEBUG
    
    #noise = np.random.random_sample(obs.shape) / conf['snr']
    #obs *= 1 + noise

    factor = max(obs[0]) / max(flux)
    flux = flux * factor
    intensities = intensities.apply(lambda s: s * factor)
    
    #wl_psg, f_psg = psg.load_flux(conf)
    #wl_si, i_psg = limb_darkening.load_intensities(
    #        conf, par, wl_psg, f_psg)
    
    #plt.plot(wl, flux, wl_psg, f_psg)
    #plt.show()

    """
    i2_psg = []
    r = np.sqrt(1-imu**2)
    i_tmp = [intensities[imu[0]] * np.pi * r[0]**2]
    for j in range(1, len(imu)):
        i_tmp.append(intensities[imu[j]] * np.pi * (r[j]**2 - r[j-1]**2))
        i2_psg.append(i_psg[imu[j]] * 2 * np.pi* np.sqrt(1-imu[j]**2))

    k = 1
    plt.plot(wl, np.sum(i_tmp, 0) /np.pi , wl_psg, f_psg)
    plt.title(imu[k])
    plt.show()
    """
    #TODO END_DEBUG

    return par, wl, flux, intensities, tell, obs, phase


def main(star, planet, lamb='auto', use_fake=False, offset=0, **kwargs):
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

    #TODO
    #psg.load_psg(conf, [180, 180.04, 180.08, 180.12, 180.16, 180.20, 180.24, 180.28, 180.32, 180.36, 180.4,
    #                       179.96, 179.92, 179.88, 179.84, 179.80, 179.76, 179.72, 179.68, 179.64, 179.6])

    # Step 1: Get Data
    print('Load data')
    par, wl, flux, intensities, tell, obs, phase = get_data(
        conf, star, planet, **kwargs)

    # Step 2: Calc Solution
    print('Calculate solution')
    sol_t = calculate(conf, par, wl, obs, tell, flux,
                      intensities, phase, lamb=lamb)

    # Step 2.5: Fake Spectrum
    if use_fake:
        print('Generate synthetic spectrum')
        fake = generate_spectrum(conf, par, wl, tell, flux, intensities, phase)
        sol_f = calculate(conf, par, wl, fake, tell, flux,
                          intensities, phase, lamb=lamb)
    else:
        fake = sol_f = None

    # Step 3: Output
    print('Plot')
    offset = 1 - max(sol_t)
    plot(conf, par, wl, obs, fake, tell, flux, sol_t + offset, sol_f)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='Extract the planetary transmittance spectrum, from one or more transit observations.')
        parser.add_argument('star', type=str, help='The observed star')
        parser.add_argument('planet', type=str, help='The letter of the planet (default=b)', nargs='?', default='b')
        parser.add_argument('-l', '--lambda', type=str, help='Regularization parameter lambda (default=auto)', default='auto', dest='lamb')
        parser.add_argument('-s', '-syn', help='Create a synthetic spectrum as comparison', action='store_true')

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
            #TODO
            print('WARNING: lambda=auto does currently not work properly')
        use_fake = args.s
    else:
        star = None
        planet = None
        use_fake = True
        #lamb = 'auto'
        lamb = 1

    #offset = 52 #13.76  # TODO offset of the solution in y direction (intensity)
    # TODO size of the atmosphere in units of planetar radii (scales and shifts the solution)
    atm_factor = 0.1
    try:
        main(star, planet, lamb=lamb, use_fake=use_fake, atm_factor=atm_factor)
    except FileNotFoundError as fnfe:
        print("Some files seem to be missing, can't complete calculation")
        print(fnfe)
