"""
Create synthetic observation spectra, when telluric, stellar flux etc are given
"""

from os.path import join
import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d as gaussbroad

import intermediary as iy
import psg

def generate_spectrum(conf, par, wl_tell, telluric, wl_flux, flux, intensity, source='psg'):
    """ Generate a fake spectrum """

    #TODO determine suitable phases independently
    max_phase = iy.maximum_phase(par)
    n_obs = 20
    phase = np.linspace(180-max_phase, 180 + max_phase, num=n_obs)
    phase = np.deg2rad(phase)

    # Sigma of Instrumental FWHM in pixels
    sigma = 1 / 2.355 * conf['fwhm']

    try:
        # Load planet spectrum
        if source in ['psg']:
            wl, planet = psg.load_input(conf, None)
    except FileNotFoundError:
        print('No planet spectrum for synthetic observation found')
        raise FileNotFoundError

    planet = gaussbroad(planet, sigma)

    # Specific intensities
    _i_planet, _i_atm = iy.specific_intensities(par, phase, intensity)

    flux = np.interp(wl, wl_flux, flux)    
    telluric = np.interp(wl, wl_tell, telluric)

    i_planet = np.zeros((len(phase), len(wl)))
    i_atm = np.zeros((len(phase), len(wl)))
    
    for i in range(len(phase)):
        i_planet[i] = np.interp(wl, wl_flux, _i_planet[i])
        i_atm[i] = np.interp(wl, wl_flux, _i_atm[i])



    # Observed spectrum
    obs = (flux[None, :] - i_planet * par['A_planet+atm'] +
           par['A_atm'] * i_atm * planet[None, :]) * telluric
    # Generate noise
    noise = np.random.randn(len(phase), len(wl)) / conf['snr']

    obs = gaussbroad(obs, sigma) * (1 + noise)
    return wl, obs, phase