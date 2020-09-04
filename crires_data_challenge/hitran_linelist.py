import pandas as pd
import numpy as np
import astropy.units as u

from os.path import dirname, join
from scipy.ndimage import gaussian_filter1d

from cats.spectrum import Spectrum1D


class Hitran:
    def __init__(self):
        fname = join(dirname(__file__), "../data/h2o.par")

        df = pd.read_table(fname)
        self.table = df
        self.wavelength = 1 / self.table["nu"]
        self.wavelength = self.wavelength << u.Unit("cm")
        self.wavelength = self.wavelength.to("AA")


class HitranSpectrum(Spectrum1D):
    def __init__(self):
        fname = join(dirname(__file__), "../data/H2O_transmission_spec-inject.dat")

        df = pd.read_table(
            fname, skiprows=1, header=None, names=["wavelength", "flux"], sep="\s+"
        )
        wavelength = df["wavelength"].values << u.Unit("cm")
        flux = df["flux"].values
        # flux = gaussian_filter1d(flux, 20)
        # flux -= np.min(flux)
        # flux /= np.max(flux)
        flux = np.sqrt(1 - flux)
        flux -= np.min(flux)
        flux = 1 - flux ** 2

        flux = flux << u.one

        super().__init__(
            spectral_axis=wavelength, flux=flux, reference_frame="barycentric"
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    hitspec = HitranSpectrum()
    plt.plot(hitspec.wavelength.to_value("AA"), hitspec.flux)
    plt.show()
