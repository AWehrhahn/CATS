"""
Load spectra from Planetary Spectrum Generator (PSG)
"""
import glob
from os.path import basename, exists, join, splitext

import numpy as np
import pandas as pd

from data_module_interface import data_module
from DataSources.PSG import PSG
from dataset import dataset


class psg(data_module):

    @classmethod
    def apply_modifiers(cls, conf, par, ds):
        """ Apply modifiers from conf to dataset ds

        Parameters:
        ----------
        conf : {dict}
            configuration setting
        par : {dict}
            stellar and planetary parameters
        ds : {dataset}
            dataset to modify
        Returns
        -------
        ds : dataset
            modified dataset
        """
        if 'psg_wl_mod' in conf.keys():
            ds.wl *= float(conf['psg_wl_mod'])
        if 'psg_flux_mod' in conf.keys():
            ds.scale *= float(conf['psg_flux_mod'])
        return ds

    @classmethod
    def load_input(cls, config, par):
        """ load planetary transmission spectrum

        Parameters:
        ----------
        config : {dict}
            configuration settings
        par : {dict}
            stellar and planetary parameters
        Returns
        -------
        planet : dataset
            planetary transmission spectrum
        """
        input_file = join(config['input_dir'],
                          config['psg_dir'], config['psg_file_atm'])

        planet = pd.read_csv(input_file)
        #TODO remove wl offset
        wl = planet['Wave/freq'].values / 10 + 5000/10000
        planet = planet['Total'].values

        ds = dataset(wl, planet)
        ds = cls.apply_modifiers(config, par, ds)

        return ds

    @classmethod
    def load_observations(cls, config, par, *args, **kwargs):
        """ load observations created by PSG

        Parameters:
        ----------
        config : {dict}
            configuration settings
        par : {dict}
            stellar and planetary parameters
        *args

        **kwargs
        Returns
        -------
        obs : dataset
            observations
        """

        """ load observations """
        obs_file = join(config['input_dir'],
                        config['psg_dir'], config['psg_file_obs'])
        phase_file = join(config['input_dir'],
                          config['psg_dir'], config['psg_file_phase'])
        phase = pd.read_table(
            phase_file, delim_whitespace=True, index_col='filename')

        # Find all suitable files
        files = glob.glob(obs_file)

        obs_all = []
        wl_all = []
        phase_all = []

        for f in files:
            obs = pd.read_csv(f)

            wl = obs['Wave/freq'].values
            obs = obs['Total'].values

            wl_all.append(wl)
            obs_all.append(obs)

            bn = basename(f)
            bn = splitext(bn)[0]

            phase_all.append(phase.loc[bn]['phase'])

        wl_all = np.array(wl_all)
        obs_all = np.array(obs_all)
        phase_all = np.array(phase_all)
        phase_all = np.deg2rad(phase_all)

        ds = dataset(wl, obs_all)
        ds = cls.apply_modifiers(config, par, ds)
        ds.phase = phase_all

        return ds

    @classmethod
    def load_stellar_flux(cls, config, par):
        """ load stellar flux generated by PSG

        Parameters:
        ----------
        config : {dict}
            configuration settings
        par : {dict}
            stellar and planetary parameters
        Returns
        -------
        stellar : dataset
            stellar flux
        """
        flux_file = join(config['input_dir'],
                         config['psg_dir'], config['psg_file_star'])
        flux = pd.read_csv(flux_file)

        wl = flux['Wave/freq'].values
        flux = flux['Stellar'].values

        wl, flux = cls.apply_modifiers(config, par, wl, flux)

        return wl, flux

    @classmethod
    def load_tellurics(cls, config, par):
        """ load telluric transmission spectrum generated by PSG

        Parameters:
        ----------
        config : {dict}
            configuration settings
        par : {dict}
            stellar and planetary parameters
        Returns
        -------
        tellurics : dataset
            telluric transmission spectrum
        """
        tell_file = join(config['input_dir'],
                         config['psg_dir'], config['psg_file_tell'])
        tell = pd.read_csv(tell_file)

        wl = tell['Wave/freq'].values
        tell = tell['Telluric'].values

        wl, tell = cls.apply_modifiers(config, par, wl, tell)

        return wl, tell

    @classmethod
    def load_psg(cls, config, phase, wl_low=0.6, wl_high=2.0, steps=140):
        """ load synthetic spectra from Planetary Spectrum Generator webservice

        PSG doesn't allow large wavelength ranges at high resolution, therefore split it into small parts that will be calculated
        Generated data is stored in the PSG input folder, defined in config
        https://psg.gsfc.nasa.gov/

        Parameters:
        ----------
        config : {dict}
            configuration settings
        phase : {float, np.ndarray}

        wl_low : {float}, optional
            lower end of the wavelength range to calculate in microns (the default is 0.6)
        wl_high : {float}, optional
            upper end of the wavelength range to calculate in microns (the default is 2.0)
        steps : {int}, optional
            number of parts that the wavelength range is split into, should be around 100 steps
            per 0.1 micron (the default is 140, which fits the default values for wl_low and wl_high)
        """
        psg_file = join(config['input_dir'],
                        config['psg_dir'], config['psg_file'])
        psg = PSG(config_file=psg_file)

        # Get telluric
        tell_file = join(config['input_dir'],
                         config['psg_dir'], config['psg_file_tell'])
        if not exists(tell_file):
            df = psg.get_data_in_range(
                wl_low, wl_high, steps, wephm='T', type='tel')
            df.to_csv(tell_file, index=False)

        # Get planet
        atm_file = join(config['input_dir'],
                        config['psg_dir'], config['psg_file_atm'])
        if not exists(atm_file):
            df = psg.get_data_in_range(
                wl_low, wl_high, steps, wephm='T', type='trn')
            df.to_csv(atm_file, index=False)

        # Get stellar flux
        flx_file = join(config['input_dir'],
                        config['psg_dir'], config['psg_file_star'])
        if not exists(flx_file):
            df = psg.get_data_in_range(wl_low, wl_high, steps, wephm='T')
            df.to_csv(flx_file, index=False)

        for i, p in enumerate(phase):
            # Get radiance
            obs_file = join(config['input_dir'], config['psg_dir'],
                            config['psg_file_obs'].replace('*', str(i)))
            if not exists(obs_file):
                psg.change_config({'OBJECT-SEASON': p})
                df = psg.get_data_in_range(wl_low, wl_high, steps, wephm='T')
                df.to_csv(obs_file, index=False)
