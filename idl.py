"""
Load data generated in IDL
"""
from os.path import join

import numpy as np
from scipy.io import readsav

from data_module_interface import data_module
from dataset import dataset


class idl(data_module):
    """ Class to load data in IDL """
    @classmethod
    def load_solar(cls, conf, par, calib_dir):
        """ Load the rdnso2011 solar spectrum as prepared by SME

        TODO: remove calib_dir parameter

        Parameters:
        ----------
        conf : {dict}
            configuration settings
        par : {dict}
            stellar and planetary parameters
        calib_dir : {str}
            directory that contains the idl file
        Returns
        -------
        solar : dataset
            The rdnso2011 spectrum
        """
        s_fname = join(calib_dir, conf['idl_file_solar'])
        data = readsav(s_fname)
        wave = data['w']
        flux = data['s']

        wave, unique = np.unique(wave, return_index=True)
        flux = flux[unique]

        ds = dataset(wave, flux)
        return ds

    @classmethod
    def load_SME(cls, conf, par):
        fname = join(conf['input_dir'], conf['idl_dir'], conf['idl_file_sme'])
        data = readsav(fname)
        sme = data['sme']
        #wavelength grid
        wave = sme.wave[0]
        # continuum modifier ?
        continuum = sme.cmod[0]

        #wavelength indices of the various sections
        wave_index = sme.wind[0]
        #observed spectrum
        obs_flux = sme.sob[0]
        return wave, wave_index, obs_flux, continuum
