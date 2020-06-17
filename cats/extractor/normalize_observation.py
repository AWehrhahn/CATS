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
from copy import deepcopy

from ..simulator.detector import Detector
from ..spectrum import SpectrumArray, SpectrumList


def continuum_normalize(spectra: SpectrumArray, blaze: np.ndarray):
    # Correct for blaze function
    # spectra /= blaze.ravel()
    # spectra = [spec / blaze for spec in tqdm(spectra)]

    # TODO Continuum normalize
    # Normalize to the same median
    # Without overlap between orders its going to be difficult to normalize
    # Maybe we can have some observations of the out of transit be in H/2/4 to fill the gaps?
    # We can't change it during transit, and the gaps are larger than the radial velocity shift
    for i in tqdm(range(len(spectra))):
        for left, right in zip(spectra.segments[:-1], spectra.segments[1:]):
            f = spectra.flux[i, left:right]
            d = np.nanpercentile(f, 95)
            spectra.flux[i, left:right] /= d

    return spectra


def continuum_normalize_part_2(
    spectra: SpectrumArray,
    stellar: SpectrumArray,
    telluric: SpectrumArray,
    detector: Detector,
):
    for j in tqdm(range(len(spectra))):
        for left, right in zip(spectra.segments[:-1], spectra.segments[1:]):
            simulation = stellar.flux[j, left:right] * telluric.flux[j, left:right]
            simulation = simulation.to_value(u.one)
            simulation = detector.apply_instrumental_broadening(simulation)

            x = spectra.wavelength[j, left:right].to_value(u.AA)
            y = spectra.flux[j, left:right].to_value(1)
            yp = simulation

            mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(yp)
            x, y, yp = x[mask], y[mask], yp[mask]
            if len(x) == 0:
                continue
            x0 = x[0]
            x -= x0

            def func(x, *c):
                return y * np.polyval(c, x)

            deg = 3
            p0 = np.ones(deg + 1)
            popt, pcov = curve_fit(func, x, yp, p0=p0)

            # For debugging
            # plt.plot(x, y * np.polyval(popt, x), label="observation")
            # plt.plot(x, yp, label="model")
            # plt.show()

            x = spectra.wavelength[j, left:right].to_value(u.AA) - x0
            spectra.flux[j, left:right] *= np.polyval(popt, x)

    return spectra


def normalize_observation(
    spectra: SpectrumArray,
    stellar: SpectrumArray,
    telluric: SpectrumArray,
    detector: Detector,
):
    spectra = deepcopy(spectra)
    sort = np.argsort(spectra.datetime)
    j = sort[len(sort) // 2]

    # TODO: Use only out of transit observations for normalization
    # TODO: Shift all observations used for the normalization to barycentric restframe
    # TODO: and resample onto the same wavelength grid
    # TODO: normalize each observation to its median first?

    norm = np.nanmedian(spectra.flux.to_value(u.one), axis=0)
    cont = np.ones_like(norm)
    for left, right in zip(spectra.segments[:-1], spectra.segments[1:]):
        f = norm[left:right]
        d = np.nanpercentile(f, 95)
        cont[left:right] *= d
        norm[left:right] /= d

    for left, right in zip(spectra.segments[:-1], spectra.segments[1:]):
        simulation = stellar.flux[j, left:right] * telluric.flux[j, left:right]
        simulation = simulation.to_value(u.one)
        simulation = detector.apply_instrumental_broadening(simulation)

        x = spectra.wavelength[j, left:right].to_value(u.AA)
        y = norm[left:right]
        yp = simulation

        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(yp)
        x, y, yp = x[mask], y[mask], yp[mask]
        if len(x) == 0:
            continue
        x0 = x[0]
        x -= x0

        def func(x, *c):
            return y * np.polyval(c, x)

        deg = 5
        p0 = np.ones(deg + 1)
        popt, pcov = curve_fit(func, x, yp, p0=p0)

        # For debugging
        # plt.plot(x, y * np.polyval(popt, x), label="observation")
        # plt.plot(x, yp, label="model")
        # plt.show()

        x = spectra.wavelength[j, left:right].to_value(u.AA) - x0
        cont[left:right] /= np.polyval(popt, x)
        norm[left:right] *= np.polyval(popt, x)

    for j in range(len(spectra)):
        spectra.flux[j] /= cont

    return spectra

    # Divide by the blaze and the median of each observation
    spectra = continuum_normalize(norm, detector.blaze)
    # Use stellar * telluric as a reference model to normalize each observation
    spectra = continuum_normalize_part_2(norm, stellar, telluric, detector)
    # spectra = SpectrumArray(spectra)
    return spectra


if __name__ == "__main__":
    from ..simulator.detector import Crires

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
