import logging
from os.path import dirname, join

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage import io
from skimage import transform as tf

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
star = "WASP-107"
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

print("SNR required:", snr)
print("Maximum exposure time", t_exp)

# Run the Runnert
# data = runner.run(["solve_problem"])

# d = data["solve_problem"]
# for k, v in d.items():
#     plt.plot(v.wavelength, v.flux, label=f"{k}")
# plt.legend()
# plt.show()


data = runner.run_module("cross_correlation", load=True)

# runner.steps["cross_correlation"].plot(data, sysrem_iterations=5, sysrem_iterations_afterwards=6)

# for i in range(3, 10):
#     plt.plot(np.sum(data[f"{i}"][10:27], axis=0) / 100, label=f"{i}")
#     for j in range(10):
#         plt.plot(np.sum(data[f"{i}.{j}"][10:27], axis=0) / 100, label=f"{i}.{j}")



plt.imshow(data["5"], aspect="auto", origin="lower")
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
