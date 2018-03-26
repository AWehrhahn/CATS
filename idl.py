"""
Load data generated in IDL
"""
import glob
from os.path import join

import numpy as np
from scipy.io import readsav
import astropy.io.fits as fits
import jdcal
from awlib.reduce import echelle

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
    def load_stellar_flux(cls, conf, par):
        wave, wave_index, obs_flux, continuum = cls.load_SME(conf, par)

        sort = np.argsort(wave)
        wave = wave[sort]
        obs_flux = obs_flux[sort]

        wave = np.ascontiguousarray(wave)
        obs_flux = np.ascontiguousarray(obs_flux)

        return dataset(wave, obs_flux)

    @classmethod
    def load_observations(cls, conf, par, *args, **kwargs):
        fname = join(conf['input_dir'], conf['idl_dir'], conf['idl_file_obs'])
        files = glob.glob(fname)
        obs = [cls.load(conf=conf, par=par, fname=g) for g in files]

        # Fix wl grid
        for i in range(1, len(obs)):
            obs[i].wl = obs[0].wl

        # Organize everything into a single dataset
        flux = np.array([ob.flux for ob in obs])
        err = np.array([ob.err for ob in obs])
        phase = np.array([ob.phase for ob in obs])

        obs = dataset(obs[0].wl, flux, err)
        obs.phase = phase

        return obs

    @classmethod
    def load(cls, conf, par, fname):
        ech = echelle.rdech(fname)
        header = fits.open(fname)[0].header

         # calc phases
        tmid = header['MJD-OBS']  # in mjd
        transit = par['transit'] - jdcal.MJD_0
        period = par['period']
        phase = ((tmid - (transit - period / 2)) / period) % 1
        phase = 360 * phase

        ds = dataset(ech.wave, ech.spec, ech.sig)
        ds.phase = phase
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
        #synthetic spectrum
        obs_flux = sme.smod[0]
        return wave, wave_index, obs_flux, continuum
