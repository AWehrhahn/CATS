import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.optimize import curve_fit
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import RegularGridInterpolator

from astropy.nddata import StdDevUncertainty
import astroplan

from tqdm import tqdm
from astropy.constants import c

from ..least_squares.least_squares import least_squares


from .datasource import DataSource
from ..spectrum import SpectrumArray


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
        tell = t0 + t1 * airmass[:, None]
        tell = np.clip(tell, 0, 1)
        stel = f + g * (fp - f)  # np.diff(f, append=2 * f[-1] - f[-2])
        obs = tell * stel
        return obs

    def fitfunc(param):
        t0 = 1
        n = param.size // 2
        t1 = param[:n]
        f = param[n:]
        fp = np.roll(f, -1)
        model = plotfunc(airmass, t0, t1, f, fp, g[:, l:r])
        resid = model - yflux[:, l:r]
        return resid.ravel()

    def regularization(param):
        n = param.size // 2
        t1 = param[:n]
        f = param[n:]
        d1 = np.gradient(t1)
        d2 = np.gradient(f)
        reg = np.concatenate((d1, d2))
        return reg ** 2

    t0 = np.ones_like(coeff[:, 1])
    t1 = coeff[:, 0] / coeff[:, 1]
    t1[(t1 > 0.1) | (t1 < -2)] = -2
    f = np.copy(coeff[:, 1])

    for k in tqdm(range(1)):
        for l, r in tqdm(
            zip(spectra.segments[:-1], spectra.segments[1:]),
            total=spectra.nseg,
            leave=False,
        ):
            n = r - l

            # Smooth first guess
            mu = gaussian_filter1d(t1[l:r], 1)
            var = gaussian_filter1d((t1[l:r] - mu) ** 2, 11)
            sig = np.sqrt(var) * 80 + 0.5
            sig = np.nan_to_num(sig)
            smax = int(np.ceil(np.nanmax(sig))) + 1
            points = [t1[l:r]] + [gaussian_filter1d(t1[l:r], i) for i in range(1, smax)]
            smooth = RegularGridInterpolator((np.arange(smax), np.arange(n)), points)(
                (sig, np.arange(n))
            )
            t1[l:r] = smooth

            # plt.plot(t1[l:r])
            # plt.plot(f[l:r])
            # plt.show()

            # fold = np.copy(f[l:r])
            # told = np.copy(t1[l:r])

            # Bounds for the optimisation
            bounds = np.zeros((2, 2 * n))
            bounds[0, :n], bounds[0, n:] = -2, 0
            bounds[1, :n], bounds[1, n:] = 0, 1
            x0 = np.concatenate((t1[l:r], f[l:r]))
            x0 = np.nan_to_num(x0)
            x0 = np.clip(x0, bounds[0], bounds[1])

            res = least_squares(
                fitfunc,
                x0,
                method="trf",
                bounds=bounds,
                max_nfev=200,
                tr_solver="lsmr",
                tr_options={"atol": 1e-2, "btol": 1e-2},
                jac_sparsity="auto",
                regularization=regularization,
                r_scale=0.2,
                verbose=2,
                diff_step=0.01,
            )
            t0[l:r] = 1
            t1[l:r] = res.x[:n]
            f[l:r] = res.x[n:]

            # lower, upper = [-2, 0, 0], [0, 1, 1]
            # for i in tqdm(range(l, r - 1), leave=False):
            #     x0 = [t1[i], f[i], f[i + 1]]
            #     x0 = np.nan_to_num(x0)
            #     x0 = np.clip(x0, lower, upper)
            #     res = least_squares(fitfunc, x0, method="trf", bounds=[lower, upper])
            #     t0[i] = 1
            #     t1[i], f[i] = res.x[0], res.x[1]

            # t0[l:r] = gaussian_filter1d(t0[l:r], 0.5)
            # t1[l:r] = gaussian_filter1d(t1[l:r], 0.5)
            # f[l:r] = gaussian_filter1d(f[l:r], 0.5)

        # total = 0
        # for i in range(len(spectra)):
        #     total += np.sum(
        #         (plotfunc(airmass[i], t0, t1, f, np.roll(f, -1), g[i]) - yflux[i]) ** 2
        #     )
        # print(total)

    # TODO: t0 should be 1 in theory, however it is not in practice because ?
    tell = t0 + t1 * airmass[:, None]
    tell = np.clip(tell, 0, 1)
    tell = tell << spectra.flux.unit

    # i = 10
    # plt.plot(wavelength, yflux[i], label="observation")
    # plt.plot(
    #     wavelength,
    #     plotfunc(airmass[i], t0, t1, f, np.roll(f, -1), g[i]),
    #     label="combined",
    # )
    # plt.plot(wavelength, tell[i], label="telluric")
    # plt.plot(wavelength, f, label="stellar")
    # plt.legend()
    # plt.show()

    flux = f + g * (np.roll(f, -1, axis=0) - f)
    # flux = np.tile(f, (len(spectra), 1))
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
        flux=tell,
        spectral_axis=wave,
        uncertainty=uncs,
        segments=spectra.segments,
        datetime=spectra.datetime,
        star=spectra.star,
        planet=spectra.planet,
        observatory_location=spectra.observatory_location,
        reference_frame="telescope",
    )

    # print("Shift observations to the telescope restframe")
    # spec = spec.shift("barycentric", inplace=True)

    # spec = spec.shift("telescope", inplace=True)
    spec = spec.resample(spectra.wavelength, method="linear", inplace=True)
    tell = tell.resample(spectra.wavelength, method="linear", inplace=True)

    return spec, tell


class CombineStellar(DataSource):
    def __init__(self, spectra, mask, telluric, detector, stellar):
        # combine
        spectra = deepcopy(spectra)
        self.combined, self.telluric = combine_observations(spectra)

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
