import numpy as np
import astropy.units as u

from astropy.io import fits
from specutils.spectra import SpectralRegion
from specutils.manipulation import extract_region

from .datasource import DataSource, StellarIntensities
from ..spectrum import Spectrum1D
from ..reference_frame import TelescopeFrame

class TelluricModel(DataSource):
    def __init__(self, fname, star, observatory):
        super().__init__()
        self.fname = fname
        self.star = star
        self.observatory = observatory

    def get(self, wave, time):
        wmin, wmax = wave.min(), wave.max()
        wrange = SpectralRegion(wmin, wmax)

        hdu = fits.open(self.fname)
        data = hdu[1].data

        data_wave = data["wave"] << u.AA
        data_flux = data["flux"] << u.one

        spec = Spectrum1D(flux=data_flux, spectral_axis=data_wave)

        spec = extract_region(spec, wrange)

        spec.__class__ == Spectrum1D
        spec.description = "telluric transmission spectrum from a model"
        spec.source = f"Evangelos/CRIRES+ wiki; {self.fname}"
        spec.datetime = time
        spec.meta["star"] = self.star
        spec.meta["observatory_location"] = self.observatory
        spec.reference_frame = "telescope"

        return spec