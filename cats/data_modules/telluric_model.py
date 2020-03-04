import numpy as np
import astropy.units as u

from astropy.io import fits
from specutils.spectra import SpectralRegion
from specutils.manipulation import extract_region

from .datasource import DataSource, StellarIntensities
from ..spectrum import Spectrum1D, SpectrumList
from ..reference_frame import TelescopeFrame


class TelluricModel(DataSource):
    def __init__(self, fname, star, observatory):
        super().__init__()
        self.fname = fname
        self.star = star
        self.observatory = observatory

    def get(self, wrange, time):
        hdu = fits.open(self.fname)
        data = hdu[1].data

        data_wave = data["wave"] << u.AA
        data_flux = data["flux"] << u.one
        spec = Spectrum1D(flux=data_flux, spectral_axis=data_wave)

        wave, flux = [], []
        for wmin, wmax in wrange.subregions:
            subrange = SpectralRegion(wmin, wmax)
            s = extract_region(spec, subrange)
            wave += [s.wavelength]
            flux += [s.flux]

        spectra = SpectrumList(
            flux=flux,
            spectral_axis=wave,
            description="telluric transmission spectrum from a model",
            source=f"Evangelos/CRIRES+ wiki; {self.fname}",
            datetime=time,
            star=self.star,
            observatory_location=self.observatory,
            reference_frame="telescope",
        )

        return spectra
