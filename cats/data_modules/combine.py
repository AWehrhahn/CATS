import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.optimize import curve_fit

from .datasource import DataSource
from ..spectrum import SpectrumArray
from ..solver.linear import LinearSolver


def detect_ouliers(spectra: SpectrumArray):
    flux = np.copy(spectra.flux)
    for i in range(len(spectra)):
        flux[i] /= np.nanpercentile(flux[i], 95)

    median = np.nanmedian(flux, axis=0)
    flux = np.abs(flux - median)
    mad = np.nanmedian(flux, axis=0)
    mask = flux > 5 * mad
    mask |= np.isnan(spectra.flux)

    flux = np.ma.array(spectra.flux, mask=mask)
    spectrum = np.ma.mean(flux, axis=0)
    uncs = np.ma.std(flux, axis=0)
    return spectrum, uncs


def combine_observations(spectra: SpectrumArray, blaze: np.ndarray):
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
    spectrum, unc = detect_ouliers(spectra)

    # Normalize to upper envelope
    print("Normalize combined spectrum")
    spectrum /= blaze.ravel()
    unc /= blaze.ravel()
    uncs = []
    for left, right in zip(spectra.segments[:-1], spectra.segments[1:]):
        lvl = np.nanpercentile(spectrum[left:right], 95)
        spectrum[left:right] /= lvl
        uncs += [unc[left:right] / lvl << u.one]

    # plt.plot(spectrum.wavelength[0], spectrum.flux[0])
    # plt.show()

    spectrum = np.ma.getdata(spectrum)
    spectrum = SpectrumArray(
        flux=spectrum[None, :],
        spectral_axis=wavelength[None, :],
        segments=spectra.segments,
        datetime=[spectra.datetime[len(spectra) // 2]],
    )

    spectrum = spectrum[0]
    return spectrum, uncs


class CombineStellar(DataSource):
    def __init__(self, spectra, blaze, mask, telluric, detector):
        self.spectra = spectra
        self.blaze = blaze
        self.mask = mask
        self.telluric = telluric

        # combine
        self.combined, self.uncs = combine_observations(self.spectra, self.blaze)
        # normalize
        for i in range(len(self.combined)):
            x = self.combined.wavelength[i][mask[i]]
            y = self.combined.flux[i][mask[i]]
            x0 = self.combined.wavelength[i][0]
            x -= x0
            x = x.to_value("AA")
            y = y.to_value()
            coeff = np.polyfit(x, y, 3)
            x = self.combined.wavelength[i] - x0
            x = x.to_value("AA")
            c = np.polyval(coeff, x)
            self.combined.flux[i] /= c << u.one

            func = lambda x, *p: np.polyval(p, x) * y

            x = self.combined.wavelength[i].to_value(u.AA)
            y = self.combined.flux[i].to_value(1)
            yp = telluric[len(telluric) // 2].flux[i].to_value(1)

            m = np.isfinite(x) & np.isfinite(y) & np.isfinite(yp)
            x, y, yp = x[m], y[m], yp[m]
            x0 = x[0]
            x -= x0

            deg = 3
            p0 = np.ones(deg + 1)
            popt, pcov = curve_fit(func, x, yp, p0=p0)

            x = self.combined.wavelength[i].to_value(u.AA) - x0
            self.combined._data[i] *= np.polyval(popt, x)

        # Disentangle
        # obs - stellar * telluric = 0
        regularization_weight = 1e-4
        self.stellar = [None for _ in self.combined]
        for i in range(len(self.combined)):
            g = self.combined.flux[i].to_value()
            f = telluric[len(telluric) // 2].flux[i].to_value()
            f = np.nan_to_num(f, nan=1.0)
            f = detector.apply_instrumental_broadening(f)
            f = detector.apply_instrumental_broadening(f)

            solver = LinearSolver(None, None, None)
            if regularization_weight is None:
                l = solver.best_lambda(f, g)
            else:
                l = regularization_weight
            self.stellar[i] = solver.Tikhonov(f, g, l)
            self.stellar[i] = self.stellar[i] << u.one

            # plt.plot(
            #     self.combined.wavelength[i], self.combined.flux[i], label="observation"
            # )
            # plt.plot(self.combined.wavelength[i], f, label="telluric")
            # plt.plot(self.combined.wavelength[i], self.stellar[i], label="stellar")
            # plt.legend()
            # plt.show()

        pass

    def get(self, wrange, time):
        spec = deepcopy(self.combined)
        spec.datetime = time
        return spec
