"""
Load data generated in IDL
"""
import glob
from os.path import join
import logging

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy import units as q
import jdcal

try:
    from pyreduce import echelle
    _has_pyreduce = True
except ImportError:
    logging.error("Need to install pyreduce, to use pyreduce module")
    _has_pyreduce = False

from .data_interface import data_observations
from .dataset import dataset

from ..orbit import Orbit


class pyreduce(data_observations):
    """ Class to load data in IDL """

    mjd0 = Time(0, format="mjd").jd

    def load(self, fname, parameters):
        ech = echelle.read(fname)
        header = ech.header

        # calc phases
        tmid = header['MJD-OBS']  # in mjd
        tmid += header["EXPTIME"] * 0.5 * q.second.to(q.day) # go to the middle of the exposure
        transit = parameters['transit'] - pyreduce.mjd0
        period = parameters['period']
        phase = ((tmid - (transit - period / 2)) / period) % 1
        phase *= 2 * np.pi

        sort = np.ma.argsort(ech["wave"])
        wave = ech["wave"][sort]
        spec = ech["spec"][sort]
        sig = ech["sig"][sort]

        ds = dataset(wave, spec, sig)
        ds.phase = phase
        return ds

    def get_observations(self, **data):
        parameters = data["parameters"]

        o = Orbit(self.configuration, parameters)

        star = self.configuration["_star"]
        planet = self.configuration["_planet"]
        input_dir = self.configuration["input_dir"].format(star=star, planet=planet)

        fname = join(input_dir, self.configuration['filename'])
        files = glob.glob(fname)
        obs = [self.load(g, parameters) for g in files]

        # Fix wl grid
        wgrid = obs[0].wave
        for i in range(1, len(obs)):
            obs[i].wave = wgrid

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
