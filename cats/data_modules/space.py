import numpy as np

import astropy.units as u

from ..spectrum import Spectrum1D, SpectrumList
from .datasource import DataSource


class Space(DataSource):
    """
    Tellurics for space based telescopes,
    i.e. No telluric absorption features,
    will just return 1
    """

    def get(self, wrange, time=None):

        wave = []
        flux = []
        for wr in wrange:
            wmin, wmax = wr.lower, wr.upper
            wave += [np.geomspace(wmin, wmax, 100) << wmax.unit]
            flux += [np.ones(100) << u.one]

        spec = SpectrumList(
            flux=flux,
            spectral_axis=wave,
            description="tellurics in space, i.e. no tellurics",
            source="space",
            reference_frame="barycentric",
        )

        return spec
