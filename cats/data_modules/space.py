import numpy as np

import astropy.units as u

from ..spectrum import Spectrum1D
from .datasource import DataSource


class Space(DataSource):
    """
    Tellurics for space based telescopes,
    i.e. No telluric absorption features,
    will just return 1
    """

    def get_tellurics(self, spec):
        wave = spec.wavelength
        flux = np.ones(wave.size) << u.Unit(1)
        ds = Spectrum1D(flux=flux, spectral_axis=wave)
        return ds
