import logging
from os.path import dirname, join

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage import io
from skimage import transform as tf
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.stats import ttest_ind, norm

from astropy import units as u
from astropy.constants import c
from astropy.utils.iers import IERS_Auto

from cats.simulator.detector import Crires
from cats.extractor.runner import CatsRunner

from exoorbit.orbit import Orbit

# TODO List:
# - automatically mask points before fitting with SME
# - if star and planet steps aren't run manually, we use the initial values
#   instead we should load the data if possible
# - Tests for all the steps
# - Refactoring of the steps, a lot of the code is strewm all over the place
# - Determine Uncertainties for each point


def shear(x, shear=1, inplace=False):
    afine_tf = tf.AffineTransform(shear=shear)
    modified = tf.warp(x, inverse_map=afine_tf)
    return modified


def gauss(x, height, mu, sig, floor):
    return height * np.exp(-(((x - mu) / sig) ** 2) / 2) + floor


def gaussfit(x, y, p0=None):
    """
    Fit a simple gaussian to data

    gauss(x, a, mu, sigma, floor) = a * exp(-z**2/2) + floor
    with z = (x - mu) / sigma

    Parameters
    ----------
    x : array(float)
        x values
    y : array(float)
        y values
    Returns
    -------
    gauss(x), parameters
        fitted values for x, fit paramters (a, mu, sigma)
    """

    if p0 is None:
        p0 = [np.max(y) - np.min(y), 0, 1, np.min(y)]

    popt, _ = curve_fit(gauss, x, y, p0=p0)
    return gauss(x, *popt), popt


# Update IERS tables if necessary
IERS_Auto()

# Detector
setting = "K/2/4"
detectors = [1, 2, 3]
orders = [7, 6, 5, 4, 3, 2]
detector = Crires(setting, detectors, orders=orders)

# Linelist
linelist = join(dirname(__file__), "crires_k_2_4.lin")

# Star info
star = "WASP 107"
planet = "b"

# Initialize the CATS runner
base_dir = join(dirname(__file__), "../datasets/WASP-107b_SNR100")
raw_dir = join(base_dir, "Spectrum_00")
medium_dir = join(base_dir, "medium")
done_dir = join(base_dir, "done")
runner = CatsRunner(
    detector,
    star,
    planet,
    linelist,
    base_dir=base_dir,
    raw_dir=raw_dir,
    medium_dir=medium_dir,
    done_dir=done_dir,
)

# Override data with known information
star = runner.star
planet = runner.planet
orbit = Orbit(star, planet)

atmosphere_height = planet.atm_scale_height(star.teff)
snr = star.radius ** 2 / (2 * planet.radius * atmosphere_height)
snr = snr.decompose()

velocity_semi_amplitude = orbit.radial_velocity_semiamplitude_planet()
t_exp = c / (2 * np.pi * velocity_semi_amplitude) * planet.period / detector.resolution
t_exp = t_exp.decompose()


print("SNR required: ", snr)
print("Maximum exposure time: ", t_exp)
print(f"Planet Velocity Kp {velocity_semi_amplitude.to('km/s')}")

# Run the Runnert
# data = runner.run(["solve_problem"])

# d = data["solve_problem"]
# for k, v in d.items():
#     plt.plot(v.wavelength, v.flux, label=f"{k}")
# plt.legend()
# plt.show()


data = runner.run_module("cross_correlation", load=True)

spectra = runner.data["spectra"]

# runner.steps["cross_correlation"].plot(data, sysrem_iterations=5, sysrem_iterations_afterwards=6)

# for i in range(3, 10):
#     plt.plot(np.sum(data[f"{i}"][10:27], axis=0) / 100, label=f"{i}")
#     for j in range(10):
#         plt.plot(np.sum(data[f"{i}.{j}"][10:27], axis=0) / 100, label=f"{i}.{j}")

data = data["7"]
rv = np.linspace(-100, 100, 201)

plt.imshow(data, aspect="auto", origin="lower")
plt.xlabel("rv [km/s]")
xticks = plt.xticks()[0][1:-1]
xticks_labels = np.interp(xticks, np.arange(len(rv)), rv)
xticks_labels = [f"{x:.3g}" for x in xticks_labels]
plt.xticks(xticks, labels=xticks_labels)
plt.show()

datetime = spectra.datetime
phi = (datetime - planet.time_of_transit) / planet.period
phi = phi.to_value(1)
# We only care about the fraction
phi = phi % 1
c_light = 3e5

interpolator = interp1d(rv, data, kind="linear", bounds_error=False)

vsys = np.linspace(-25, 25, 101)
kp = np.linspace(0, 150, 300)
combined = np.zeros((len(kp), len(vsys)))
for i, vs in enumerate(vsys):
    for j, k in enumerate(kp):
        vp = vs + k * np.sin(2 * np.pi * phi)
        # shifted = [np.interp(vp[i], rv, data[i], left=np.nan, right=np.nan) for i in range(len(vp))]
        shifted = np.diag(interpolator(vp))
        combined[j, i] = np.nansum(shifted)

plt.imshow(combined, aspect="auto", origin="lower")

plt.xlabel("vsys [km/s]")
xticks = plt.xticks()[0][1:-1]
xticks_labels = np.interp(xticks, np.arange(len(vsys)), vsys)
xticks_labels = [f"{x:.3g}" for x in xticks_labels]
plt.xticks(xticks, labels=xticks_labels)

plt.ylabel("Kp [km/s]")
yticks = plt.yticks()[0][1:-1]
yticks_labels = np.interp(yticks, np.arange(len(kp)), kp)
yticks_labels = [f"{y:.3g}" for y in yticks_labels]
plt.yticks(yticks, labels=yticks_labels)

plt.show()

plt.subplot(211)
mean = np.nanmean(combined, axis=0)
vsys_peak = 65 #np.argmax(mean)
curve, vsys_popt = gaussfit(
    vsys[vsys_peak - 10 : vsys_peak + 10],
    mean[vsys_peak - 10 : vsys_peak + 10],
    p0=[mean[vsys_peak] - np.min(mean), vsys[vsys_peak], 1, np.min(mean)],
)
plt.plot(vsys, mean)
plt.vlines(vsys[vsys_peak], np.min(mean), mean[vsys_peak], "k", "--")
plt.plot(vsys, gauss(vsys, *vsys_popt), "r--")
plt.xlabel("vsys [km/s]")
plt.subplot(212)
peak = combined[:, vsys_peak - 1 : vsys_peak + 2]
mean = np.nanmean(peak, axis=1)
kp_peak = np.argmax(mean)
curve, kp_popt = gaussfit(
    kp,
    mean,
    p0=[mean[kp_peak] - np.min(mean), kp[kp_peak], 1, np.min(mean)],
)
plt.plot(kp, mean)
plt.vlines(kp[kp_peak], np.min(mean), mean[kp_peak], "k", "--")
plt.plot(kp, gauss(kp, *kp_popt), "r--")
plt.xlabel("Kp [km/s]")
plt.show()



# Have to check that this makes sense
in_trail = combined[:, vsys_peak - 3 : vsys_peak + 5]
out_trail = np.hstack([combined[:, :vsys_peak - 3], combined[:, vsys_peak + 5:]])

hrange = (np.min(combined), np.max(combined))
bins = 100
in_values, hbins = np.histogram(in_trail.ravel(), bins=bins, range=hrange, density=True)
out_values, _ = np.histogram(out_trail.ravel(), bins=bins, range=hrange, density=True)

tresult = ttest_ind(in_trail.ravel(), out_trail.ravel(), equal_var=False)
sigma = norm.isf(tresult.pvalue)

plt.hist(in_trail.ravel(), bins=hbins, density=True, histtype="step")
plt.hist(out_trail.ravel(), bins=hbins, density=True, histtype="step")
plt.show()


# plt.imshow(data["5.6"], aspect="auto", origin="lower")
# plt.show()

# plt.plot(np.sum(shear(data[f"5.6"], -0.8), axis=0) / 100, label=f"5.6")


# plt.xlabel("v [km/s]")
# xticks = plt.xticks()[0][1:-1]
# xticks_labels = xticks - 100
# plt.xticks(xticks, labels=xticks_labels)

# plt.ylabel("ccf [SNR]")

# plt.legend()
# plt.show()

pass
