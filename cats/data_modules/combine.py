import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.optimize import curve_fit

from astropy.nddata import StdDevUncertainty

from .datasource import DataSource
from ..spectrum import SpectrumArray
from ..extractor.normalize_observation import continuum_normalize_part_2


def detect_ouliers(spectra: SpectrumArray):
    flux = np.copy(spectra.flux)
    for i in range(len(spectra)):
        for j, k in zip(spectra.segments[:-1], spectra.segments[1:]):
            flux[i, j:k] /= np.nanpercentile(flux[i, j:k], 95)

    median = np.nanmedian(flux, axis=0)
    flux = np.abs(flux - median)
    mad = np.nanmedian(flux, axis=0)
    mad *= 1.4826  # to scale to gaussian sigma
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
    spectra = spectra.resample(wavelength, inplace=True, method="linear")
    # Detects ouliers based on the Median absolute deviation
    mask = detect_ouliers(spectra)

    # Average the spectra
    flux = np.copy(spectra.flux)
    flux[mask] = np.nan
    lvl = [
        [None for _ in range(len(spectra.segments) - 1)] for _ in range(len(spectra))
    ]
    for i in range(len(spectra)):
        for j, (left, right) in enumerate(
            zip(spectra.segments[:-1], spectra.segments[1:])
        ):
            lvl[i][j] = np.nanpercentile(flux[i, left:right], 95)
            flux[i, left:right] /= lvl[i][j]

    spectrum = np.nanmean(flux, axis=0)
    uncs = np.nanstd(flux, axis=0)
    uncs = StdDevUncertainty(uncs, copy=False)

    # spectra2 += spectrum[None, :]
    # # for i in range(len(spectra)):
    # #     for j, (left, right) in enumerate(zip(spectra.segments[:-1], spectra.segments[1:])):
    # #         spectra2[i, left:right] = spectrum[left:right] * lvl[i][j]

    spec = SpectrumArray(
        flux=spectrum[None, :],
        spectral_axis=wavelength[None, :],
        uncertainty=uncs[None, :],
        segments=spectra.segments,
        datetime=spectra.datetime[:1],
        reference_frame="barycentric",
    )

    return spec


class CombineStellar(DataSource):
    def __init__(self, spectra, mask, telluric, detector, stellar):
        # combine
        spectra = deepcopy(spectra)
        self.combined = combine_observations(spectra)

        # Repeat the data for each observation
        nobs = len(spectra)

        # The NDUncertainty class doesn't like being handled like a np.ndarray
        # so we give it some special treatment
        unc_cls = self.combined.uncertainty.__class__
        uncs = self.combined.uncertainty.array
        uunit = self.combined.uncertainty.unit
        uncs = np.tile(uncs, (nobs, 1)) << uunit
        uncs = unc_cls(uncs)

        self.combined = SpectrumArray(
            flux=np.tile(self.combined.flux, (nobs, 1)),
            spectral_axis=np.tile(self.combined.wavelength, (nobs, 1)),
            uncertainty=uncs,
            segments=spectra.segments,
            datetime=spectra.datetime,
            reference_frame="barycentric",
        )

        self.combined = continuum_normalize_part_2(
            self.combined, stellar, telluric, detector
        )
        pass

    def get(self, wrange, time):
        idx = self.combined.datetime == time
        idx = np.where(idx)[0][0]
        spec = deepcopy(self.combined[idx])
        return spec
