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
        """ Load the solar spectrum prepared by rdnso2011 in IDL/SME """
        s_fname = join(calib_dir, conf['idl_file_solar'])
        data = readsav(s_fname)
        wave = data['w']
        flux = data['s']

        wave, unique = np.unique(wave, return_index=True)
        #wave = wave[unique]
        flux = flux[unique]

        ds = dataset(wave, flux)
        return ds
