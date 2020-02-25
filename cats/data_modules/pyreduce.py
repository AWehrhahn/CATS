"""
Load data generated in IDL
"""
import glob
import os.path
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
    from pyreduce.configuration import get_configuration_for_instrument
    from pyreduce.reduce import main

    _has_pyreduce = True
except ImportError:
    logging.error("Need to install pyreduce, to use pyreduce module")
    _has_pyreduce = False

from .data_interface import data_observations, data_reduction
from .dataset import dataset

from ..orbit import Orbit

c = speed_of_light * 1e-3


class pyreduce(data_observations, data_reduction):
    """ Class to load data in IDL """

    _requires = ["parameters"]

    def load(self, fname, parameters, order=1):
        ech = echelle.read(fname, continuum_normalization=False)
        header = ech.header

        wave = ech["wave"][order]
        spec = ech["spec"][order]
        sig = ech["sig"][order]

        sort = np.ma.argsort(wave, axis=None)
        wave = wave.flat[sort].compressed()
        spec = spec.flat[sort].compressed()
        sig = sig.flat[sort].compressed()

        mask = (spec > 0) & (spec < np.nanpercentile(spec, 99)) & (~np.isnan(spec)) & (wave != 0)
        wave = wave[mask]
        spec = spec[mask]
        sig = sig[mask]

        spec /= header["EXPTIME"]
        # spec /= np.median(spec)

        # calc phases
        # go to the middle of the exposure
        tmid = float(header["MJD-OBS"])  # in mjd
        tmid = tmid + header["EXPTIME"] * 0.5 * q.second.to(q.day)
        phase = self.orbit.get_phase(tmid)

        ds = dataset(wave, spec, sig)
        ds.phase = phase
        ds.time = tmid
        ds.header = header
        return ds

    @staticmethod
    def _shift(rv, wgrid, wave, flux, reference):
        w = wgrid * (1 - rv / c)
        f = interp1d(wave, flux, bounds_error=False, fill_value=(flux[0], flux[-1]))(w)
        return f - reference

    def get_observations(self, **data):
        parameters = data["parameters"]
        self.orbit = Orbit(self.configuration, parameters)

        star = self.configuration["_star"]
        planet = self.configuration["_planet"]
        telescope = self.configuration["telescope"]
        instrument = self.configuration["instrument"]
        input_dir = (
            os.path.join(self.configuration["base_dir"], self.configuration["output_dir"])
        ).format(star=star, planet=planet, instrument=instrument, telescope=telescope)

        if "reduction" in data.keys():
            files = data["reduction"]
        else:
            fname = join(input_dir, self.configuration["filename"])
            files = glob.glob(fname)
        obs = [self.load(g, parameters) for g in files]

        # Split observations into different days
        times = np.empty(len(obs))
        for i in range(len(obs)):
            times[i] = obs[i].time
        # days = np.round(times + 0.5)
        # unique = np.unique(days)

        wgrid = obs[0].wave
        for i in range(1, len(obs)):
            obs[i].new_grid(wgrid)
        # for day in unique:
        #     mask = days == day
        #     points = np.arange(len(obs))[mask]
        #     i = points[0]
        #     reference = obs[i].data / np.median(obs[i].data)
        #     reference = interp1d(
        #         obs[i].wave, reference, fill_value=0, bounds_error=False
        #     )(wgrid)
        #     for i in points:
        #         # Cross-correlate spectra, so lines align
        #         wave = obs[i].wave
        #         flux = obs[i].data / np.median(obs[i].data)

        #         corr = correlate(flux, reference[reference != 0], mode="same")
        #         offset = np.argmax(corr)
        #         rv = (wave[len(wave) // 2] / wave[offset] - 1) * c

        #         res = least_squares(
        #             pyreduce._shift,
        #             rv,
        #             args=(wgrid, wave, flux, reference),
        #             loss="soft_l1",
        #         )
        #         rv = res.x

        #         obs[i]._wave_orig = wave * (1 + rv / c)
        #         obs[i].new_grid(wgrid)

        # Organize everything into a single dataset
        flux = np.empty((len(obs), len(obs[0].data)))
        err = np.empty((len(obs), len(obs[0].data)))
        phase = np.empty(len(obs))

        for i in range(len(obs)):
            flux[i] = obs[i].data
            err[i] = obs[i].error
            phase[i] = obs[i].phase

        # Identify observation runs (days)
        # Normalize each day
        # TODO: Normalize to lightcurve according to the phase?
        # for day in unique:
        #     flux[days == day] /= np.median(flux[days == day])

        # DEBUG Remove specific days
        # mask = (days != unique[0]) & (days != unique[-1]) & (days != unique[1]) & (days != unique[-2])
        # mask = days == unique[2]

        # flux = flux[mask]
        # err = err[mask]
        # times = times[mask]
        # phase = phase[mask]
        # other = other[mask]

        obs = dataset(wgrid, flux, err)
        obs.time = times
        obs.phase = phase
        return obs

    def get_reduced(self, **data):
        if "raw" in data.keys():
            instrument, mode, target, folder = data["raw"]
            base_dir = os.path.join(folder, "..")
            input_dir = os.path.basename(folder)
            output_dir = "reduced"
        else:
            instrument = self.configuration["instrument"]
            mode = self.configuration["mode"]
            target = self.configuration["_star"] + self.configuration["_planet"]
            base_dir = self.configuration["base_dir"].format(**self.configuration)
            input_dir = self.configuration["input_dir"].format(**self.configuration)
            output_dir = self.configuration["output_dir"].format(**self.configuration)

        night = "????-??-??"
        config = get_configuration_for_instrument(instrument, plot=False)
        steps = self.configuration["steps"]

        output = main(
            instrument,
            target,
            night,
            mode,
            steps=steps,
            base_dir=base_dir,
            input_dir=input_dir,
            output_dir=output_dir,
            configuration=config,
        )
        return output[0]["finalize"]

