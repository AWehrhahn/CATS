"""
Load data generated in IDL
"""
import glob
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import readsav
import astropy.io.fits as fits
import jdcal
from awlib.reduce import echelle

from data_module_interface import data_module
from dataset import dataset


class reduce(data_module):
    """ Class to load data in IDL """

    @classmethod
    def load_stellar_flux(cls, conf, par):
        cls.log(2, 'REDUCE')
        fname = join(conf['input_dir'], conf['harps_dir'],
                     conf['reduce_file_stellar'])
        stellar = cls.load(conf, par, fname)
        return stellar

    @classmethod
    def load_observations(cls, conf, par, *args, **kwargs):
        cls.log(2, 'REDUCE')
        fname = join(conf['input_dir'], conf['harps_dir'],
                     conf['reduce_file_obs'])
        files = glob.glob(fname)
        obs = [cls.load(conf=conf, par=par, fname=g) for g in files]

        # Fix wl grid
        # TODO is an equidistant wavelength grid necessary?
        # obs[0].wl = np.linspace(obs[0].wl[0], obs[0].wl[-1], num=len(obs[0].wl))
        for i in range(1, len(obs)):
            obs[i].wl = obs[0].wl

        # Organize everything into a single dataset
        flux = np.empty((len(obs), len(obs[0].flux[0])))
        err = np.empty((len(obs), len(obs[0].err[0])))
        phase = np.empty(len(obs))

        for i in range(len(obs)):
            flux[i] = obs[i].flux[0]
            err[i] = obs[i].err[0]
            phase[i] = obs[i].phase

        obs = dataset(obs[0].wl, flux, err)
        obs.phase = phase

        return obs

    @classmethod
    def load(cls, conf, par, fname, colrange=None):
        ech = echelle.rdech(fname)
        header = fits.open(fname)[0].header
        if colrange is None:
            sav = join(conf['input_dir'], conf['harps_dir'],
                       'harps_red.ord_norm.sav')
            sav = readsav(sav)
            colrange = sav['col_range']

        # calc phases
        tmid = header['MJD-OBS']  # in mjd
        transit = par['transit'] - jdcal.MJD_0
        period = par['period']
        phase = ((tmid - (transit - period / 2)) / period) % 1
        phase *= 2 * np.pi

        mask = np.full(ech.wave.shape, False, dtype=bool)
        for i in range(len(colrange)):
            mask[i, colrange[i, 0]:colrange[i, 1] + 1] = True

        wave = ech.wave[mask]
        spec = ech.spec[mask]
        sig = ech.sig[mask]

        sort = np.argsort(wave)
        wave = wave[sort]
        spec = spec[sort]
        sig = sig[sort]

        ds = dataset(wave, spec, sig)
        # ds.gaussbroad(5) #TODO questionable at least
        ds.phase = phase
        return ds
