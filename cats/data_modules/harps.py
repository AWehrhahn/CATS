"""
Load reduced HARPS observations
"""

import glob
import logging
from os.path import join

import astropy.io.fits as fits
import jdcal
import joblib
import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d as gaussbroad
from scipy.optimize import minimize

from awlib.astro import air2vac, doppler_shift, planck
from awlib.reduce.echelle import rdech
from awlib.util import normalize

from . import orbit as orb
from .data_interface import data_observation, data_tellurics
from .data_module_interface import data_module
from .dataset import dataset
from .idl import idl
from .marcs import marcs


class harps(data_observation, data_tellurics):
    """ access HARPS data
    """

    def load(self, fname):
        """ load a single FITS file with HARPS observations

        Assumes that the primary header contains a table with three columns
        WAVE : the wavelength
        FLUX : the spectrum
        ERR : the errors on the spectrum
        as well as TMID in its header, which is the julian date at the middle of the observation

        Parameters:
        ----------
        fname : {str}
            filename, relative to harps directory as defined in conf
        apply_barycentric : {bool}, optional
            apply barycentric correction if True (the default is False)

        Returns
        -------
        obs : dataset
            a single HARPS observation, including the orbital phase
        """
        fname = join(self.configuration["input_dir"], self.configuration["harps_dir"], fname)
        hdulist = fits.open(fname)
        data = hdulist[1].data
        header = hdulist[1].header

        wave = data["WAVE"][0]
        flux = data["FLUX"][0]
        err = data["ERR"][0]

        wave = air2vac(obs.wl)
        obs = dataset(wave, flux, err)
        obs.time = header["TMID"]
        # Barycentric correction
        prime_header = hdulist[0].header
        obs.rv += -prime_header["ESO DRS BERV"]

        return obs

    def get_observations(self, **_):
        """ Load all observations from all fits files in the HARPS input directory

        The HARPS input directory is defined in conf

        Returns
        -------
        observation : dataset of shape (nobs, nwave)
            Observations
        """
        logging.info("Get observations from HARPS")
        fname = join(
            self.configuration["input_dir"],
            self.configuration["harps_dir"],
            self.configuration["harps_file_obs"],
        )

        # Load data
        obs = [self.load(g) for g in glob.glob(fname)]

        # TODO: Organize all observations into a single object

        return obs

    def get_tellurics(self):
        """ load telluric transmission spectrum

        The spectrum is taken from the ESO SkyCalc online tool
        http://www.eso.org/observing/etc/bin/gen/form?INS.MODE=swspectr+INS.NAME=SKYCALC

        Parameters:
        ----------
        conf : {dict}
            configuration settings
        par : {dict}
            stellar and planetary parameters
        Returns
        -------
        telluric : dataset
            telluric transmission spectrum
        """
        logging.info("Get tellurics from HARPS")
        fname = join(self.configuration["input_dir"], self.configuration["harps_dir"], self.configuration["harps_file_tell"])
        df = pd.read_table(fname, delim_whitespace=True)
        wl = df["wave"].values * 10
        tell = df["tell"].values
        ds = dataset(wl, tell)

        return ds
