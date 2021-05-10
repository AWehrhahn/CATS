import logging
from os.path import dirname, join

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage import io
from skimage import transform as tf
from scipy.interpolate import interp1d

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

data = data["5"]
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

vsys = np.linspace(-100, 100, 401)
kp = np.linspace(0, 201, 401)
combined = np.zeros((len(vsys), len(kp)))
for i, vs in enumerate(vsys):
    for j, k in enumerate(kp):
        vp = vs + k * np.sin(2 * np.pi * phi)
        # shifted = [np.interp(vp[i], rv, data[i], left=np.nan, right=np.nan) for i in range(len(vp))]
        shifted = np.diag(interpolator(vp))
        combined[i, j] = np.nansum(shifted)

plt.imshow(combined, aspect="auto", origin="lower")

plt.xlabel("Kp [km/s]")
xticks = plt.xticks()[0][1:-1]
xticks_labels = np.interp(xticks, np.arange(len(kp)), kp)
xticks_labels = [f"{x:.3g}" for x in xticks_labels]
plt.xticks(xticks, labels=xticks_labels)

plt.ylabel("vsys [km/s]")
yticks = plt.yticks()[0][1:-1]
yticks_labels = np.interp(yticks, np.arange(len(vsys)), vsys)
yticks_labels = [f"{y:.3g}" for y in yticks_labels]
plt.yticks(yticks, labels=yticks_labels)

plt.show()

plt.imshow(data["5.6"], aspect="auto", origin="lower")
plt.show()

plt.plot(np.sum(shear(data[f"5.6"], -0.8), axis=0) / 100, label=f"5.6")


plt.xlabel("v [km/s]")
xticks = plt.xticks()[0][1:-1]
xticks_labels = xticks - 100
plt.xticks(xticks, labels=xticks_labels)

plt.ylabel("ccf [SNR]")

plt.legend()
plt.show()

pass
