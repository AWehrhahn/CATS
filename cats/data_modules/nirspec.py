import os.path
import numpy as np
from astropy.io import fits
import glob

from scipy.optimize import least_squares

from .data_interface import data_observations, data_stellarflux
from .dataset import dataset

class nirspec(data_observations, data_stellarflux):

    _obs_requires = ["parameters"]
    _flux_requires = ["parameters", "observations"]

    def load(self, fname):
        hdu = fits.open(fname)
        header = hdu[0].header
        data = hdu[1].data

        wave = data["wave (A)"]
        flux = data["flux (cnts)"]
        error = data["noise (cnts)"]

        ds = dataset(wave, flux, error)

        fname2 = fname.replace("fitstbl", "fits").replace("flux_tbl", "flux")
        header = fits.open(fname2)[0].header

        ds.time = header["MJD-OBS"]

        return ds


    def get_observations(self, **_):
        order = "34"
        star = self.configuration["_star"]
        planet = self.configuration["_planet"]
        fname = self.configuration["input_dir"].format(star=star, planet=planet)
        fname = os.path.join(fname, "extracted", "fitstbl", "flux",
            f"NS.????????.?????_{order}_flux_tbl.fits.gz",
        )
        fname = os.path.abspath(fname)

        obs = [self.load(g) for g in glob.glob(fname)]

        # Wavelength calibration is not good enough to align spectra properly
        # Use cross-correlation? Maybe PyReduce will be good enough?
        # For each spectrum (past the first) find the offset that minimizes the difference
        data = obs[0].data
        for i in range(1, len(obs)):
            def func(x):
                wave = obs[i].wave * (1 + x/3e5)
                shifted = obs[i]._interpolate(obs[i].data, obs[i].wave, wave)[0]
                return shifted - data

            res = least_squares(func, 0)        
            rv = res.x[0]
            obs[i].shift(None, rv)
        
        nobs = len(obs)
        nwave = len(obs[0])
        # TODO find good overall wavelength frame
        wave = obs[0].wave
        datacube = np.zeros((nobs, nwave))
        dates = np.zeros(nobs)
        for i in range(nobs):
            datacube[i] = obs[i]._interpolate(obs[i].data, obs[i].wave, wave)[0]
            dates[i] = obs[i].time

        obs = dataset(wave, datacube)
        obs.time = dates
        return obs

    def get_stellarflux(self, **data):
        obs = data["observations"]
        stellar = np.median(obs.data, axis=0)
        stellar = dataset(obs.wave, stellar)
        return stellar
