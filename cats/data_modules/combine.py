import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.optimize import curve_fit

from .datasource import DataSource
from ..spectrum import SpectrumArray
from ..extractor.normalize_observation import continuum_normalize_part_2

def detect_ouliers(spectra: SpectrumArray):
    flux = np.copy(spectra.flux)
    for i in range(len(spectra)):
        flux[i] /= np.nanpercentile(flux[i], 95)

    median = np.nanmedian(flux, axis=0)
    flux = np.abs(flux - median)
    mad = np.nanmedian(flux, axis=0)
    mask = flux > 5 * mad
    mask |= np.isnan(spectra.flux)

    return mask


def combine_observations(spectra: SpectrumArray):
    # TODO: The telluric spectrum will change between observations
    # and therefore influence the recovered stellar parameters
    # Especially when we combine data from different transits!

    # for i in range(spectra.shape[0]):
    #     plt.plot(spectra.wavelength[i], spectra.flux[i], "r")

    # Shift to the same reference frame (barycentric)
    print("Shift observations to the barycentric restframe")
    spectra = spectra.shift("barycentric", inplace=True)

    # Arbitrarily choose the central grid as the common one
    print("Combine all observations")
    wavelength = spectra.wavelength[len(spectra) // 2]
    spectra = spectra.resample(wavelength)
    # Detects ouliers based on the Median absolute deviation
    mask = detect_ouliers(spectra)

    # Average the spectra
    flux = np.ma.array(np.copy(spectra.flux), mask=mask)
    lvl = [[None for _ in range(len(spectra.segments) - 1)] for _ in range(len(spectra))]
    for i in range(len(spectra)):
        for j, (left, right) in enumerate(zip(spectra.segments[:-1], spectra.segments[1:])):
            lvl[i][j] = np.nanpercentile(flux[i, left:right], 95)
            flux[i, left:right] /= lvl[i][j]

    spectrum = np.ma.mean(flux, axis=0)
    spectrum = np.ma.getdata(spectrum)
    uncs = np.ma.std(flux, axis=0)

    spectra2 = np.zeros((len(spectra), spectrum.size)) << spectrum.unit
    wavelength2 = np.zeros((len(spectra), spectrum.size)) << wavelength.unit
    wavelength2 += wavelength[None, :]

    spectra2 += spectrum[None, :]
    # for i in range(len(spectra)):
    #     for j, (left, right) in enumerate(zip(spectra.segments[:-1], spectra.segments[1:])):
    #         spectra2[i, left:right] = spectrum[left:right] * lvl[i][j]

    spec = SpectrumArray(
        flux=spectra2,
        spectral_axis=wavelength2,
        segments=spectra.segments,
        datetime=spectra.datetime,
        reference_frame="barycentric",
    )

    return spec


class CombineStellar(DataSource):
    def __init__(self, spectra, mask, telluric, detector, stellar):
        # combine
        spectra = deepcopy(spectra)
        self.combined = combine_observations(spectra)
        self.combined = continuum_normalize_part_2(self.combined, stellar, telluric, detector)
        pass

    def get(self, wrange, time):
        idx = self.combined.datetime == time
        idx = np.where(idx)[0][0]
        spec = deepcopy(self.combined[idx])
        return spec
