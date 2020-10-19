import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.optimize import curve_fit, least_squares
from scipy.ndimage.filters import gaussian_filter1d

from astropy.nddata import StdDevUncertainty
import astroplan

from tqdm import tqdm
from astropy.constants import c


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

    # Shift to the same reference frame (telescope)
    print("Shift observations to the telescope restframe")
    spectra = spectra.shift("telescope", inplace=True)

    # Arbitrarily choose the central grid as the common one
    print("Combine all observations")
    wavelength = spectra.wavelength[len(spectra) // 2]
    spectra = spectra.resample(wavelength, inplace=True, method="linear")
    # Detects ouliers based on the Median absolute deviation
    mask = detect_ouliers(spectra)

    # TODO: other approach
    # s(i) = f(i) (1 - w / dw * g) + f(i+1) w / dw * g
    # g = sqrt((1 + beta) / (1 - beta)) - 1
    rv = np.zeros(len(spectra))
    for i in range(len(spectra)):
        rv[i] = spectra.reference_frame.to_barycentric(spectra.datetime[i]).to_value(
            "km/s"
        )
    rv -= np.mean(rv)
    rv *= -1
    rv /= c.to_value("km/s")
    rv = np.sqrt((1 + rv) / (1 - rv)) - 1

    wave = np.copy(wavelength.to_value("AA"))
    for l, r in zip(spectra.segments[:-1], spectra.segments[1:]):
        wave[l:r] /= np.gradient(wave[l:r])

    g = wave[None, :] * rv[:, None]

    # TODO: the tellurics are scaled by the airmass, which we should account for here, when creating the master stellar
    # TODO: Could we fit a linear polynomial to each wavelength point? as a function of time/airmass?
    # TODO: Then the spectrum would be constant, since there is no change, but for tellurics we would see a difference
    # TODO: But there is a different offset for the tellurics, and the stellar features
    yflux = spectra.flux.to_value(1)
    flux = np.zeros(yflux.shape)
    coeff = np.zeros((yflux.shape[1], 2))
    airmass = spectra.airmass
    mask = ~mask
    for i in tqdm(range(spectra.flux.shape[1])):
        coeff[i] = np.polyfit(airmass[mask[:, i]], yflux[:, i][mask[:, i]], 1)
        flux[:, i] = np.polyval(coeff[i], airmass)

    # fitfunc = lambda t0, t1, f: (t0[None, :] + t1[None, :] * airmass[:, None]) * (
    #     f + g * np.diff(f, append=f[-2])
    # )

    def plotfunc(airmass, t0, t1, f, fp, g):
        tell = t0 + t1 * airmass
        tell = np.clip(tell, 0, 1)
        stel = f + g * (fp - f)  # np.diff(f, append=2 * f[-1] - f[-2])
        obs = tell * stel
        return obs

    def fitfunc(param):
        # progress.update(1)
        t0 = 1
        t1, f, fp = param
        # n = r - l
        # t0 = param[:n]
        # t1 = param[n : 2 * n]
        # f = param[2 * n :]
        # fp = np.roll(f, -1)
        model = plotfunc(airmass, t0, t1, f, fp, g[:, i])
        resid = model - yflux[:, i]
        # regularization = np.abs(f - fp)
        return resid.ravel()

    t0 = np.ones_like(coeff[:, 1])
    t1 = coeff[:, 0] / coeff[:, 1]
    t1[(t1 > 0.1) | (t1 < -2)] = -2
    f = np.copy(coeff[:, 1])

    for k in tqdm(range(2)):
        for l, r in tqdm(
            zip(spectra.segments[:-1], spectra.segments[1:]),
            total=spectra.nseg,
            leave=False,
        ):
            # Bounds for the optimisation
            lower, upper = [-2, 0, 0], [0, 1, 1]
            for i in tqdm(range(l, r - 1), leave=False):
                x0 = [t1[i], f[i], f[i + 1]]
                x0 = np.nan_to_num(x0)
                x0 = np.clip(x0, lower, upper)
                res = least_squares(fitfunc, x0, method="trf", bounds=[lower, upper])
                t0[i] = 1
                t1[i], f[i] = res.x[0], res.x[1]

            # t0[l:r] = gaussian_filter1d(t0[l:r], 0.5)
            t1[l:r] = gaussian_filter1d(t1[l:r], 0.5)
            f[l:r] = gaussian_filter1d(f[l:r], 0.5)

        total = 0
        for i in range(len(spectra)):
            total += np.sum(
                (plotfunc(airmass[i], t0, t1, f, np.roll(f, -1), g[i]) - yflux[i]) ** 2
            )
        print(total)

    # TODO: t0 should be 1 in theory, however it is not in practice because ?
    tell = t0 + t1 * airmass[:, None]
    tell = np.clip(tell, 0, 1)
    tell = tell << spectra.flux.unit


    i = 10
    plt.plot(wavelength, yflux[i], label="observation")
    plt.plot(wavelength, plotfunc(airmass[i], t0, t1, f, np.roll(f, -1), g[i]), label="combined")
    plt.plot(wavelength, tell[i], label="telluric")
    plt.plot(wavelength, f, label="stellar")
    plt.legend()
    plt.show()

    
    flux = np.tile(f, (len(spectra), 1))
    flux = flux << spectra.flux.unit
    wave = np.tile(wavelength, (len(spectra), 1)) << spectra.wavelength.unit
    uncs = np.nanstd(flux, axis=0)
    uncs = np.tile(uncs, (len(spectra), 1))
    uncs = StdDevUncertainty(uncs, copy=False)

    spec = SpectrumArray(
        flux=flux,
        spectral_axis=wave,
        uncertainty=uncs,
        segments=spectra.segments,
        datetime=spectra.datetime,
        star=spectra.star,
        planet=spectra.planet,
        observatory_location=spectra.observatory_location,
        reference_frame="telescope",
    )

    tell = SpectrumArray(
        flux = tell,
        spectral_axis=wave,
        uncertainty=uncs,
        segments=spectra.segments,
        datetime =spectra.datetime,
        star=spectra.star,
        planet=spectra.planet,
        observatory_location=spectra.observatory_location,
        reference_frame="telescope"
    )

    # print("Shift observations to the telescope restframe")
    # spec = spec.shift("barycentric", inplace=True)

    # spec = spec.shift("telescope", inplace=True)
    spec = spec.resample(spectra.wavelength, method="linear", inplace=True)
    tell = spec.resample(spectra.wavelength, method="linear", inplace=True)


    return spec, tell


class CombineStellar(DataSource):
    def __init__(self, spectra, mask, telluric, detector, stellar):
        # combine
        spectra = deepcopy(spectra)
        self.combined, self.telluric = combine_observations(spectra)

        # # Repeat the data for each observation
        # nobs = len(spectra)

        # # The NDUncertainty class doesn't like being handled like a np.ndarray
        # # so we give it some special treatment
        # unc_cls = self.combined.uncertainty.__class__
        # uncs = self.combined.uncertainty.array
        # uunit = self.combined.uncertainty.unit
        # uncs = np.tile(uncs, (nobs, 1)) << uunit
        # uncs = unc_cls(uncs)

        # self.combined = SpectrumArray(
        #     flux=self.combined.flux,
        #     spectral_axis=np.tile(self.combined.wavelength, (nobs, 1)),
        #     uncertainty=uncs,
        #     segments=spectra.segments,
        #     datetime=spectra.datetime,
        #     reference_frame="barycentric",
        # )

        # self.combined = continuum_normalize_part_2(
        #     self.combined, stellar, telluric, detector
        # )
        pass

    def get(self, wrange, time):
        idx = self.combined.datetime == time
        idx = np.where(idx)[0][0]
        spec = deepcopy(self.combined[idx])
        return spec

    def get_telluric(self, wrange, time):
        idx = self.telluric.datetime == time
        idx = np.where(idx)[0][0]
        spec = deepcopy(self.telluric[idx])
        return spec
