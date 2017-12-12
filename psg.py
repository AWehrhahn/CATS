"""
Load spectra from Planetary Spectrum Generator (PSG)
"""
from os.path import join, exists, basename, splitext
import glob

import numpy as np
import pandas as pd
from DataSources.PSG import PSG


def load_input(config, wl_grid):
    """ load input spectrum """
    input_file = join(config['input_dir'], config['file_atm_psg'])

    planet = pd.read_csv(input_file)
    wl = planet['Wave/freq'].values 
    planet = planet['Total'].values

    if 'psg_wl_mod' in config.keys():
        wl *= float(config['psg_wl_mod'])

    if 'psg_atm_mod' in config.keys():
        planet *= float(config['psg_atm_mod'])

    return np.interp(wl_grid, wl, planet)

def load_observation(config, n_exposures='all'):
    """ load observations """
    obs_file = join(config['input_dir'], config['file_obs_psg'])
    phase_file = join(config['input_dir'], config['file_phase_psg'])
    phase = pd.read_table(phase_file, delim_whitespace=True, index_col='filename')

    if n_exposures == 'all':
        #Find all suitable files
        files = glob.glob(obs_file)
    else:
        files = [obs_file.replace('*', i) for i in range(n_exposures)]

    obs_all = []
    wl_all = []
    phase_all = []

    for f in files:
        obs = pd.read_csv(f)

        wl = obs['Wave/freq'].values
        obs = obs['Total'].values

        if 'psg_wl_mod' in config.keys():
            wl *= float(config['psg_wl_mod'])

        if 'psg_obs_mod' in config.keys():
            obs *= float(config['psg_obs_mod'])

        wl_all.append(wl)
        obs_all.append(obs)

        bn = basename(f)
        bn = splitext(bn)[0]

        phase_all.append(phase.loc[bn]['phase'])
        
    wl_all = np.array(wl_all)
    obs_all = np.array(obs_all)
    phase_all = np.array(phase_all)

    return wl_all, obs_all, phase_all

def load_flux(config):
    """ load flux """
    flux_file = join(config['input_dir'], config['file_star_psg'])
    flux = pd.read_csv(flux_file)

    wl = flux['Wave/freq'].values
    flux = flux['Stellar'].values

    if 'psg_wl_mod' in config.keys():
        wl *= float(config['psg_wl_mod'])

    if 'psg_flux_mod' in config.keys():
        flux *= float(config['psg_flux_mod'])

    return wl, flux

def load_tellurics(config):
    """ load tellurics """
    tell_file = join(config['input_dir'], config['file_tell_psg'])
    tell = pd.read_csv(tell_file)

    wl = tell['Wave/freq'].values
    tell = tell['Telluric'].values

    if 'psg_wl_mod' in config.keys():
        wl *= float(config['psg_wl_mod'])

    if 'psg_tell_mod' in config.keys():
        tell *= float(config['psg_tell_mod'])

    return wl, tell

def load_psg(config, phase):
    """ load synthetic spectra from Planetary Spectrum Generator webservice """
    psg_file = join(config.input_dir, config['file_psg'])
    psg = PSG(config_file=psg_file)

    # Get telluric
    tell_file = join(config.input_dir, config['file_telluric'])
    if not exists(tell_file):
        df = psg.get_data_in_range(0.6, 1.0, 50, wephm='T', type='tel')
        df.to_csv(tell_file, index=False)

    # Get planet
    atm_file = join(config.input_dir, config['file_atmosphere'])
    if not exists(atm_file):
        df = psg.get_data_in_range(0.6, 1.0, 50, wephm='T', type='trn')
        df.to_csv(atm_file, index=False)

    for i, p in enumerate(phase):
        # Get radiance
        obs_file = join(config.input_dir, config['file_observation'].format(i))
        if not exists(obs_file):
            psg.change_config({'OBJECT-SEASON': p})
            df = psg.get_data_in_range(0.6, 1.0, 50, wephm='T')
            df.to_csv(obs_file, index=False)