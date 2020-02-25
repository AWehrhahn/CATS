import os.path
import numpy as np
import pandas as pd
import spectres

from ..orbit import Orbit as orbit_calculator
from .dataset import dataset
from .data_interface import data_intensities, data_stellarflux

class marcs(data_intensities, data_stellarflux):

    _flux_requires = ["parameters"]

    def get_stellarflux(self, **data):
        # TODO get parameters from data["parameters"]
        # and round to nearest grid value ?
        par = data["parameters"]
        teff = 3000
        logg = 5
        monh = 0.5
        vt = 2

        geom = "p" # or s for spherical
        mass = 0 # only has a value for spherical
        # alpha, C, N, O abundance
        a, c, n, o = 0, 0, 0, 0
        r, s = 0, 0

        fname = f"{geom}{teff:04d}_g{logg:+1.1f}_m{mass:1.1f}_t{vt:02d}_st_z{monh:+1.2f}_a{a:+1.2f}_c{c:+1.2f}_n{n:+1.2f}_o{o:+1.2f}_r{r:+1.2f}_s{s:+1.2f}.flx"
        flux_file = os.path.join(self.configuration["dir"], fname)
        wl_file = os.path.join(self.configuration['dir'], 'flx_wavelengths.vac')

        flux = pd.read_csv(flux_file, header=None, names=['flx'], sep="\t")
        wave = pd.read_csv(wl_file, header=None, names=['wave'], sep="\t")
        flux = flux.values.ravel()
        wave = wave.values.ravel()

        flux /= flux.max()

        ds = dataset(wave, flux)
        return ds
