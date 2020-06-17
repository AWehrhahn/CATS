from os.path import dirname, join
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from cats.spectrum import Spectrum1D


def load_planet():
    data_dir = join(dirname(__file__), "../data")
    fname = join(data_dir, "spectra0023IntFluxPlanetEarthContinum.bin")

    # dtype = np.dtype([("f8") * 6])
    data = np.fromfile(fname, dtype="f8")
    data = data.reshape((-1, 6))

    wavelength = data[:, 0] << u.nm
    intensity_sun = data[:, 1]
    flux_sun = data[:, 2]
    planetary_atmosphere_absorption = data[:, 3] << u.one
    earth_atmosphere_absorption = data[:, 4]
    flux_normalization = data[:, 5]

    spec = Spectrum1D(flux=planetary_atmosphere_absorption, spectral_axis=wavelength)
    return spec


if __name__ == "__main__":
    spec = load_planet()
    plt.plot(spec.wavelength, spec.flux)
    plt.show()
