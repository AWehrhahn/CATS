"""
Load MARCS model stellar spectra
marcs.astro.uu.se
"""
from os.path import join

import numpy as np
import pandas as pd

from scipy.constants import c, pi


def scaling_factor(par):
    return 1
    dist = 1/(par['parallax'] * 1e-3)  * 3.086e16 #in m
    h = 10**2 #m**2 #Telescope size
    return h / dist**2


def load_flux(config, par):
    """ load MARCS flux files """
    # Use the same stellar spectrum for all observations
    flux_file = join(config['input_dir'], config['dir_marcs'], config['file_star_marcs'])
    wl_file = join(config['input_dir'], config['dir_marcs'], config['file_wl_marcs'])

    flux = pd.read_table(flux_file, delim_whitespace=True, header=None, comment='#').values[:, 0]
    wl = pd.read_table(wl_file, delim_whitespace=True, header=None, comment='#').values[:, 0]

    if 'marcs_flux_mod' in config.keys():
        flux *= config['marcs_flux_mod']
    if 'marcs_wl_mod' in config.keys():
        wl *= config['marcs_wl_mod']

    flux *= scaling_factor(par)
    #Doppler shift
    #v = par['radial_velocity']
    #shift = (1 + v/c) * wl
    #flux = np.interp(wl, shift, flux)

    return wl, flux