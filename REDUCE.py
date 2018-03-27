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
        fname = join(conf['input_dir'], conf['harps_dir'], conf['reduce_file_stellar'])
        stellar = cls.load(conf, par, fname)
        return stellar

    @classmethod
    def load_observations(cls, conf, par, *args, **kwargs):
        cls.log(2, 'REDUCE')
        fname = join(conf['input_dir'], conf['harps_dir'], conf['reduce_file_obs'])
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
    def load(cls, conf, par, fname, colrange=None):
        ech = echelle.rdech(fname)
        header = fits.open(fname)[0].header
        if colrange is None:
            sav = join(conf['input_dir'], conf['harps_dir'], 'harps_red.ord_norm.sav')
            sav = readsav(sav)
            colrange = sav['col_range']

        # calc phases
        tmid = header['MJD-OBS']  # in mjd
        transit = par['transit'] - jdcal.MJD_0
        period = par['period']
        phase = ((tmid - (transit - period / 2)) / period) % 1
        phase *= 2 * np.pi

        n_orders = len(ech.wave)
        wl_range = sum(colrange[:, 1] - colrange[:, 0] +1)
        wave = np.zeros(wl_range)
        spec = np.zeros(wl_range)
        sig = np.zeros(wl_range)

        # flatten arrays
        k = 0
        for i in range(n_orders):
            j = colrange[i, 1] - colrange[i, 0] + 1
            wave[k:k + j] = ech.wave[i, colrange[i, 0]:colrange[i, 1]+1]
            spec[k:k + j] = ech.spec[i, colrange[i, 0]:colrange[i, 1]+1]
            sig[k:k + j] = ech.sig[i, colrange[i, 0]:colrange[i, 1]+1]
            k += j

        sort = np.argsort(wave)
        wave = wave[sort]
        spec = spec[sort]
        sig = sig[sort]

        ds = dataset(wave, spec, sig)
        ds.phase = phase
        return ds

