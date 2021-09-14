import logging
from os.path import dirname, join, realpath

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astroplan as ap
from scipy.constants import c as c_light_ms

from tqdm import tqdm

from skimage import io
from skimage import transform as tf
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.stats import ttest_ind, norm, t

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

def welch_t(a, b, ua=None, ub=None):
    # t = (mean(a) - mean(b)) / sqrt(std(a)**2 + std(b)**2)
    if ua is None:
        ua = a.std() / np.sqrt(a.size)
    if ub is None:
        ub = b.std() / np.sqrt(b.size)

    xa = a.mean()
    xb = b.mean()
    t = (xa - xb) / np.sqrt(ua**2 + ub**2)
    return t


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
star = "WASP-107"
planet = "b"

# Initialize the CATS runner
dataset = "WASP-107b_SNR200"
base_dir = realpath(join(dirname(__file__), f"../datasets/{dataset}"))
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
rv_step = 0.25
rv_range = 200
runner.configuration["cross_correlation"]["rv_range"] = rv_range
runner.configuration["cross_correlation"]["rv_points"] = int((2 * rv_range + 1) / rv_step)
runner.configuration["cross_correlation_reference"]["rv_range"] = rv_range
runner.configuration["cross_correlation_reference"]["rv_points"] = int((2 * rv_range + 1) / rv_step)


# Override data with known information
star = runner.star
planet = runner.planet
orbit = Orbit(star, planet)

planet.radius = 1 * u.Rjup
planet.mass = 1 * u.Mjup

atmosphere_height = planet.atm_scale_height(star.teff)
snr = star.radius ** 2 / (2 * planet.radius * atmosphere_height)
snr = snr.decompose()

velocity_semi_amplitude = orbit.radial_velocity_semiamplitude_planet()
t_exp = c / (2 * np.pi * velocity_semi_amplitude) * planet.period / detector.resolution
t_exp = t_exp.decompose()

print("SNR required: ", snr)
print("Maximum exposure time: ", t_exp)
print(f"Planet Velocity Kp {velocity_semi_amplitude.to('km/s')}")

# Run the Runner
# data = runner.run(["solve_problem"])
# d = data["solve_problem"]
# for k, v in d.items():
#     plt.plot(v.wavelength, v.flux, label=f"{k}")
# plt.legend()
# plt.show()

# data = runner.run_module("cross_correlation_reference", load=False)
data = runner.run_module("cross_correlation", load=True)
spectra = runner.data["spectra"]

# Barycentric correction
observer = ap.Observer.at_site("paranal")
obstime = spectra.datetime[len(spectra)//2]
sky_location = star.coordinates
sky_location.obstime = obstime
sky_location.location = observer.location
correction = sky_location.radial_velocity_correction()

# runner.steps["cross_correlation"].plot(data, sysrem_iterations=5, sysrem_iterations_afterwards=6)

# for i in range(3, 10):
#     plt.plot(np.sum(data[f"{i}"][10:27], axis=0) / 100, label=f"{i}")
#     for j in range(10):
#         plt.plot(np.sum(data[f"{i}.{j}"][10:27], axis=0) / 100, label=f"{i}.{j}")

data = data["7"]
config = runner.configuration["cross_correlation"]
rv_range = config["rv_range"]
rv_points = config["rv_points"]
rv_step = (2 * rv_range + 1) / rv_points
rv = np.linspace(-rv_range, rv_range, rv_points)

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
c_light = c_light_ms * 1e-3

interpolator = interp1d(rv, data, kind="linear", bounds_error=False)

vsys_min, vsys_max = 0, 25
kp_min, kp_max = 0, 300
vsys = np.linspace(vsys_min, vsys_max, int((vsys_max-vsys_min+1)//rv_step))
kp = np.linspace(kp_min, kp_max, int((kp_max-kp_min+1)//rv_step))
combined = np.zeros((len(kp), len(vsys)))
for i, vs in enumerate(tqdm(vsys)):
    for j, k in enumerate(tqdm(kp, leave=False)):
        vp = vs + k * np.sin(2 * np.pi * phi)
        # shifted = [np.interp(vp[i], rv, data[i], left=np.nan, right=np.nan) for i in range(len(vp))]
        shifted = np.diag(interpolator(vp))
        combined[j, i] = np.nansum(shifted)

# Normalize to the number of input spectra
combined /= data.shape[0]
combined /= combined.std()


# Normalize to median 0
median = np.nanmedian(combined)
combined -= median

kp_peak = combined.shape[0] // 2
kp_width = kp_peak

for i in range(3):
    # Determine the peak position in vsys and kp
    kp_width_int = int(np.ceil(kp_width))
    mean_vsys = np.nanmean(combined[kp_peak-kp_width_int+1:kp_peak+kp_width_int+1, :], axis=0)
    vsys_peak = np.argmax(mean_vsys)

    # And then fit gaussians to determine the width
    curve, vsys_popt = gaussfit(
            vsys,
            mean_vsys,
            p0=[mean_vsys[vsys_peak] - np.min(mean_vsys), vsys[vsys_peak], 1, np.min(mean_vsys)],
    )
    vsys_width = vsys_popt[2] / rv_step


    # Do the same for the planet velocity
    vsys_width_int = int(np.ceil(vsys_width)) // 4
    peak = combined[:, vsys_peak - vsys_width_int : vsys_peak + vsys_width_int + 1]
    mean_kp = np.nanmean(peak, axis=1)
    kp_peak = np.argmax(mean_kp)

    curve, kp_popt = gaussfit(
        kp,
        mean_kp,
        p0=[mean_kp[kp_peak] - np.min(mean_kp), kp[kp_peak], 1, np.min(mean_kp)],
    )
    kp_width = kp_popt[2] / rv_step

# Plot the results
ax = plt.subplot(121)
plt.imshow(combined, aspect="auto", origin="lower")
ax.add_patch(plt.Rectangle((vsys_peak-vsys_width, kp_peak-kp_width), 2 * vsys_width, 2 * kp_width, fill=False, color="red"))

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

plt.subplot(222)
plt.plot(vsys, gauss(vsys, *vsys_popt), "r--")
plt.plot(vsys, mean_vsys)
plt.vlines(vsys[vsys_peak], np.min(mean_vsys), mean_vsys[vsys_peak], "k", "--")
plt.xlabel("vsys [km/s]")

plt.subplot(224)
plt.plot(kp, mean_kp)
plt.vlines(kp[kp_peak], np.min(mean_kp), mean_kp[kp_peak], "k", "--")
plt.plot(kp, gauss(kp, *kp_popt), "r--")
plt.xlabel("Kp [km/s]")

plt.suptitle(dataset)
plt.show()



# Have to check that this makes sense
vsys_width = int(np.ceil(vsys_width))
kp_width = int(np.ceil(kp_width))

mask = np.full(combined.shape, False)
mask[kp_peak-kp_width:kp_peak+kp_width, vsys_peak - vsys_width : vsys_peak + vsys_width] = True

in_trail = combined[mask].ravel()
out_trail = combined[~mask].ravel()

hrange = (np.min(combined), np.max(combined))
bins = 100
in_values, hbins = np.histogram(in_trail, bins=bins, range=hrange, density=True)
out_values, _ = np.histogram(out_trail, bins=bins, range=hrange, density=True)

tresult = ttest_ind(in_trail, out_trail, equal_var=False, trim=0.25)
# TODO: What is the degrees of freedom
# If we use the number of points like in the scipy function we get very large sigma values
# so is it vsys and kp and err?
df = 3
pvalue = t.sf(np.abs(tresult.statistic), df)
sigma = norm.isf(pvalue)
# sigma = norm.isf(tresult.pvalue)

# Alternative sigma value, based on my understanding of sigmas and Gaussian distributions
# sigma = np.abs((in_trail.mean() - out_trail.mean()) / (in_trail.std() + out_trail.std()))


plt.hist(in_trail.ravel(), bins=hbins, density=True, histtype="step", label="in transit")
plt.hist(out_trail.ravel(), bins=hbins, density=True, histtype="step", label="out of transit")
plt.legend()
plt.title(f"{dataset}\nsigma: {sigma}")
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
