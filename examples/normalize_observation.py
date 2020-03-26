"""
Normalize the observation
"""
from glob import glob
from os.path import dirname, join

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from scipy.optimize import curve_fit
from tqdm import tqdm

from cats.simulator.detector import Crires
from cats.spectrum import SpectrumArray, SpectrumList


def continuum_normalize(spectra, blaze):
    # Correct for blaze function
    spectra = [spec / blaze for spec in tqdm(spectra)]

    # TODO Continuum normalize
    # Normalize to the same median
    # Without overlap between orders its going to be difficult to normalize
    # Maybe we can have some observations of the out of transit be in H/2/4 to fill the gaps?
    # We can't change it during transit, and the gaps are larger than the radial velocity shift
    for i, spec in tqdm(enumerate(spectra)):
        for j, s in enumerate(spec):
            f = s.flux.to_value(u.one)
            d = np.nanpercentile(f, 95)
            spectra[i][j]._data /= d

    return spectra


def continuum_normalize_part_2(spectra, stellar, telluric, detector):
    for j in tqdm(range(len(spectra))):
        spec = spectra[j]
        simulation = stellar[j] * telluric[j]
        simulation = detector.apply_instrumental_broadening(simulation)

        for i in tqdm(range(len(simulation))):
            x = spec[i].wavelength.to_value(u.AA)
            y = spec[i].flux.to_value(1)
            yp = simulation[i].flux.to_value(1)

            mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(yp)
            x, y, yp = x[mask], y[mask], yp[mask]
            if len(x) == 0:
                continue
            x0 = x[0]
            x -= x0

            def func(x, *c):
                return y * np.polyval(c, x)

            deg = 1
            p0 = np.ones(deg + 1)
            popt, pcov = curve_fit(func, x, yp, p0=p0)

            # For debugging
            # plt.plot(x, y * np.polyval(popt, x), label="observation")
            # plt.plot(x, yp, label="model")
            # plt.show()

            x = spec[i].wavelength.to_value(u.AA) - x0
            spectra[j][i]._data *= np.polyval(popt, x)

    return spectra


def normalize_observation(spectra, stellar, telluric, detector):
    # Divide by the blaze and the median of each observation
    spectra = continuum_normalize(spectra, detector.blaze)
    # Use stellar * telluric as a reference model to normalize each observation
    spectra = continuum_normalize_part_2(spectra, stellar, telluric, detector)
    return spectra


if __name__ == "__main__":
    data_dir = join(dirname(__file__), "noise_1", "raw")
    target_dir = join(dirname(__file__), "noise_1", "medium")
    files = join(data_dir, "*.fits")

    linelist = f"{data_dir}/crires_h_1_4.lin"

    detector = Crires("H/1/4", [1, 2, 3])
    observatory = detector.observatory
    wrange = detector.regions

    print("Loading data...")
    spectra = SpectrumArray.read(join(target_dir, "spectra.npz"))
    stellar = SpectrumArray.read(join(target_dir, "stellar.npz"))
    telluric = SpectrumArray.read(join(target_dir, "telluric.npz"))

    print("Normalizing spectra...")
    spectra = normalize_observation(spectra, stellar, telluric, detector)

    print("Saving normalized data...")
    spectra = SpectrumArray(spectra)
    spectra.write(join(target_dir, "spectra_normalized.npz"))
