"""
Load MARCS model stellar spectra
marcs.astro.uu.se
"""
from os.path import join

import pandas as pd


def load_flux(config):
    """ load MARCS flux files """
    # Use the same stellar spectrum for all observations
    flux_file = join(config['input_dir'], config['file_star_marcs'])
    wl_file = join(config['input_dir'], config['file_wl_marcs'])

    flux = pd.read_table(flux_file, delim_whitespace=True, header=None, comment='#').values[:, 0]
    wl = pd.read_table(wl_file, delim_whitespace=True, header=None, comment='#').values[:, 0]

    if 'marcs_flux_mod' in config.keys():
        flux *= config['marcs_flux_mod']
    if 'marcs_wl_mod' in config.keys():
        wl *= config['marcs_wl_mod']

    return wl, flux