"""
Load data generated in IDL
"""
import glob
from os.path import join
import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import speed_of_light
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from scipy.signal import correlate
from astropy.io import fits
from astropy.time import Time
from astropy import units as q

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

    _requires = ["parameters"]

    def load(self, fname, parameters):
        ech = echelle.read(fname, continuum_normalization=False)
        header = ech.header

        order = 1
        wave = ech["wave"][order]
        spec = ech["spec"][order]
        sig = ech["sig"][order]

        sort = np.ma.argsort(wave, axis=None)
        wave = wave.flat[sort].compressed()
        spec = spec.flat[sort].compressed()
        sig = sig.flat[sort].compressed()

        mask = (spec > 0) & (spec < np.nanpercentile(spec, 99)) & (~np.isnan(spec))
        wave = wave[mask]
        spec = spec[mask]
        sig = sig[mask]

        spec /= header["ITIME"] * header["COADD"]
        # spec /= np.median(spec)

        # calc phases
        tmid = header["MJD-OBS"]  # in mjd
        tmid += (
            header["E_EXPTIM"] * 0.5 * q.second.to(q.day)
        )  # go to the middle of the exposure
        phase = self.orbit.get_phase(tmid)

        ds = dataset(wave, spec, sig)
        ds.phase = phase
        ds.time = tmid
        ds.header = header
        return ds

    def get_observations(self, **data):
        parameters = data["parameters"]
        self.orbit = Orbit(self.configuration, parameters)

        star = self.configuration["_star"]
        planet = self.configuration["_planet"]
        input_dir = self.configuration["input_dir"].format(star=star, planet=planet)

        fname = join(input_dir, self.configuration["filename"])
        files = glob.glob(fname)
        obs = [self.load(g, parameters) for g in files]

        c = speed_of_light * 1e-3

        def shift(rv, wgrid, wave, flux):
            w = wgrid * (1 - rv / c)
            f = interp1d(
                wave, flux, bounds_error=False, fill_value=(flux[0], flux[-1])
            )(w)
            return f - reference

        # Fix wl grid
        # TODO find best reference
        # or resample onto a new wavelength grid that covers all points
        rvs = []
        i = 0
        wgrid = obs[i].wave
        reference = obs[i].data / np.median(obs[i].data)
        for i in range(len(obs)):
            # Cross-correlate spectra, so lines align
            wave = obs[i].wave
            flux = obs[i].data / np.median(obs[i].data)

            corr = correlate(flux, reference[reference != 0], mode="same")
            offset = np.argmax(corr)
            rv = (wave[len(wave) // 2] / wave[offset] - 1) * c

            res = least_squares(shift, rv, args=(wgrid, wave, flux), loss="soft_l1")
            rv = res.x
            obs[i]._wave_orig = wave * (1 + rv / c)

            rvs.append(rv)
            obs[i].new_grid(wgrid)

        # Organize everything into a single dataset
        flux = np.empty((len(obs), len(obs[0].data)))
        err = np.empty((len(obs), len(obs[0].data)))
        phase = np.empty(len(obs))
        times = np.empty(len(obs))
        other = np.empty(len(obs), dtype="U20")

        for i in range(len(obs)):
            flux[i] = obs[i].data
            err[i] = obs[i].error
            phase[i] = obs[i].phase
            times[i] = obs[i].time
            other[i] = str(obs[i].header["E_INPUT"])

        # Identify observation runs (days)
        # Normalize each day
        # TODO: Normalize to lightcurve according to the phase?
        days = np.round(times + 0.5)
        unique = np.unique(days)
        for day in unique:
            flux[days == day] /= np.median(flux[days == day])

        # DEBUG Remove specific days
        # mask = (days != unique[0]) & (days != unique[-1]) & (days != unique[1]) & (days != unique[-2])
        mask = days == unique[2]

        flux = flux[mask]
        err = err[mask]
        times = times[mask]
        phase = phase[mask]
        other = other[mask]

        obs = dataset(wgrid, flux, err)
        obs.time = times
        obs.phase = phase
        return obs
