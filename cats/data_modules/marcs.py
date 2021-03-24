import logging
from os.path import dirname, join

import numpy as np
import pandas as pd
from astropy import units as u

from .datasource import DataSource
from ..spectrum import Spectrum1D

logger = logging.getLogger(__name__)


class MarcsStellar(DataSource):

    flux_units = u.erg / u.cm ** 2 / u.s / u.AA

    def __init__(self, star):
        super().__init__()
        self.star = star
        self.data_dir = self.config["data_dir"]


    def get(self):
        # TODO Interpolate from grid
        
        teff = self.star.teff
        logg = self.star.logg
        monh = self.star.monh
        vt = self.star.vturb

        geom = "p"  # or s for spherical
        mass = 0  # only has a value for spherical
        # alpha, C, N, O abundance
        a, c, n, o = 0, 0, 0, 0
        r, s = 0, 0


        fname = f"{geom}{teff:04d}_g{logg:+1.1f}_m{mass:1.1f}_t{vt:02d}_st_z{monh:+1.2f}_a{a:+1.2f}_c{c:+1.2f}_n{n:+1.2f}_o{o:+1.2f}_r{r:+1.2f}_s{s:+1.2f}.flx"
        flux_file = join(self.data_dir, fname)
        wl_file = join(self.data_dir, "flx_wavelengths.vac")

        flux = pd.read_csv(flux_file, header=None, names=["flx"], sep="\t")
        wave = pd.read_csv(wl_file, header=None, names=["wave"], sep="\t")
        flux = flux.values.ravel() << self.flux_units
        wave = wave.values.ravel() << u.AA

        spec = Spectrum1D(
            flux=flux,
            spectral_axis=wave,
            reference_frame="barycentric",
            star=self.star,
            source="marcs",
            description=f"stellar spectrum of {self.star['name']}",
        )

        return spec
