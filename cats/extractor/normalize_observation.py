"""
Normalize the observation
"""
from glob import glob
from os.path import dirname, join

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from tqdm import tqdm
from copy import deepcopy

from scipy.ndimage import gaussian_filter1d


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
    stellar_broadening: float,
    telluric_broadening: float,
):
    sold = deepcopy(spectra)
    spectra = deepcopy(spectra)
    sort = np.argsort(spectra.datetime)

    # plt.imshow(spectra.flux, aspect="auto", origin="lower")
    # plt.xlabel("Wavelength")
    # plt.ylabel("Obs Num")
    # plt.show()

    # Normalize against the average spectrum, which will not normalize anything
    # deg = 2
    # p0 = np.array([0] * deg + [-1], dtype=float)
    # func = lambda x, *c: y * np.polyval(c, x)
    # avg_spec = np.nanmean(spectra.flux.to_value(1), axis=0)
    # for j in tqdm(range(len(spectra)), leave=False, desc="Spectrum"):
    #     for left, right in tqdm(
    #         zip(spectra.segments[:-1], spectra.segments[1:]),
    #         leave=False,
    #         desc="Segment",
    #         total=spectra.nseg,
    #     ):
    #         x = avg_spec[left:right]
    #         y = spectra.flux[j, left:right].to_value(1)
    #         n = np.arange(len(x))
    #         popt, pcov = curve_fit(func, n, x, p0=p0, method="trf", loss="soft_l1",)
    #         spectra.flux[j, left:right] *= np.polyval(popt, n)

    # Normalizing in time direction, which is not what we actually want to do
    # time = spectra.datetime.mjd
    # time -= time[len(time) // 2]
    # coeff = np.zeros((spectra.flux.shape[1], 3))
    # for i in tqdm(range(spectra.flux.shape[1]), leave=False, desc="Wavelength"):
    #     coeff[i] = np.polyfit(time, spectra.flux[:, i], 2)

    # # TODO: consider the neighbouring elements
    # # In this case we try smoothing with a cubic spline fit
    # # for i in tqdm(range(3), leave=False, desc="Smooth"):
    # #     for left, right in tqdm(
    # #         zip(spectra.segments[:-1], spectra.segments[1:]),
    # #         leave=False,
    # #         desc="Segment",
    # #         total=spectra.nseg,
    # #     ):
    # #         c = coeff[left:right, i]
    # #         n = np.arange(len(c))
    # #         coeff[left:right, i] = UnivariateSpline(n, c, s=1)(n)

    # for i in tqdm(range(spectra.flux.shape[1]), leave=False, desc="Wavelength"):
    #     values = np.polyval(coeff[i], time)
    #     # We cut off very small values which occur in teluric lines
    #     # TODO: what is the threshold? Also this creates some discontinuity which is problematic
    #     # By setting it 1, we make sure that we do not amplify anything, which is good
    #     values[values < 1] = 1
    #     spectra.flux[:, i] /= values

    for j in tqdm(range(len(spectra)), leave=False, desc="Spectrum"):
        wave = spectra.wavelength[j].to_value(u.AA)
        norm = spectra.flux[j].to_value(u.one)
        cont = np.ones_like(norm)
        for left, right in tqdm(
            zip(spectra.segments[:-1], spectra.segments[1:]),
            leave=False,
            desc="Segment",
            total=spectra.nseg,
        ):

            sflux = gaussian_filter1d(
                stellar.flux[j, left:right].to_value(u.one), stellar_broadening
            )
            tflux = gaussian_filter1d(
                telluric.flux[j, left:right].to_value(u.one), telluric_broadening
            )
            simulation = sflux * tflux

            # simulation = detector.apply_instrumental_broadening(simulation)

            x = wave[left:right]
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
            p0 = np.zeros(deg + 1)
            p0[-1] = 1 / np.mean(y)
            popt, pcov = curve_fit(func, x, yp, p0=p0)

            # For debugging
            # plt.plot(x, y * np.polyval(popt, x), label="observation")
            # plt.plot(x, yp, label="model")
            # plt.show()

            xs = wave[left:right] - x0
            value = np.polyval(popt, xs)
            spectra.flux[j, left:right] *= value
            if spectra.uncertainty is not None:
                spectra.uncertainty.array[j, left:right] *= value

    # Remove airmass signal
    # TODO: this should be the time, not the obs number
    # x = np.arange(spectra.shape[0])
    # for left, right in tqdm(
    #     zip(spectra.segments[:-1], spectra.segments[1:]),
    #     leave=False,
    #     desc="Segment",
    #     total=spectra.nseg,
    # ):
    #     y = np.nanmean(spectra.flux[:, left:right].to_value(1), axis=1)
    #     yf = np.polyval(np.polyfit(x, y, 5), x) / np.max(y)
    #     spectra.flux[:, left:right] /= yf[:, None]
    #     # TODO: where did the planet transit go?
    #     xf = np.nansum(sold.flux[:, left:right].to_value(1), axis=1) / yf
    #     xf /= np.nanpercentile(xf, 99)
    #     mask = np.full(len(x), True)
    #     mask[15:-15] = False
    #     coeff = np.polyfit(x[mask], xf[mask], 3)
    #     xf /= np.polyval(coeff, x)
    #     spectra.flux[:, left:right] *= xf[:, None]

    # This removes the telluric signal variation?
    # Like the curve, but do we want that?
    # I don't think so
    # x = np.arange(spectra.shape[0])
    # y = np.nanmean(spectra.flux.to_value(1), axis=1)
    # yf = np.polyval(np.polyfit(x, y, 5), x) / np.max(y)
    # spectra.flux /= yf[:, None]

    # The planet transit is normalized away
    # This reinserts it, which is going to be difficult in real data
    # as we have variations in the seeing, exposure time, etc.
    # xf = np.nansum(sold.flux.to_value(1), axis=1) / yf
    # xf /= np.nanpercentile(xf, 99)

    # mask = np.full(len(x), True)
    # mask[15:-15] = False

    # coeff = np.polyfit(x[mask], xf[mask], 3)
    # xf /= np.polyval(coeff, x)

    # spectra.flux *= xf[:, None]

    # Plot again
    # vmin, vmax = np.nanpercentile(spectra.flux.to_value(1), (5, 95))
    # vmin, vmax = 0.90, 1
    # plt.imshow(spectra.flux, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    # plt.xlabel("Wavelength")
    # plt.ylabel("Obs Num")
    # plt.show()

    return spectra

    # # Divide by the blaze and the median of each observation
    # spectra = continuum_normalize(norm, detector.blaze)
    # # Use stellar * telluric as a reference model to normalize each observation
    # spectra = continuum_normalize_part_2(norm, stellar, telluric, detector)
    # # spectra = SpectrumArray(spectra)
    # return spectra


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
