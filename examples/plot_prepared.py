import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.time import Time
from astropy.constants import c

from cats.simulator.detector import Crires
from cats.data_modules.stellar_db import StellarDb
from exoorbit.orbit import Orbit

detector = Crires("H/1/4", [1, 2, 3])
sdb = StellarDb()
star = sdb.get("HD209458")
planet = star.planets["b"]
orbit = Orbit(star, planet)

transit_time = "2020-05-25T10:31:25.418"
transit_time = Time(transit_time, format="fits")
planet.time_of_transit = transit_time

wave = np.load("wave.npy")
times = np.load("times.npy")
img_spectra = np.load("spectra.npy")
img_telluric = np.load("telluric.npy")
img_stellar = np.load("stellar.npy")

times = Time(times, format="fits")

simulation = img_stellar * img_telluric
simulation = np.nan_to_num(simulation)
simulation = gaussian_filter1d(simulation, detector.spectral_broadening, axis=1)

# TODO Fit planet transit orbit to this curve5
pa = orbit.phase_angle(times)
idx = np.where(orbit.mu(times) > 0)[0][[0, -1]]
y = np.nanmedian(img_spectra / np.nanmean(img_spectra, axis=0), axis=1)
plt.plot(pa, y)
plt.vlines(pa[idx], np.min(y), np.max(y))
plt.show()

# TODO: plot the expected position of a spectral line on top
rv = orbit.radial_velocity_planet(times)
central_wavelength = 15000

beta = (rv / c).to_value(1)
shifted = np.sqrt((1 + beta) / (1 - beta)) * central_wavelength

plt.imshow(img_spectra / simulation, aspect="auto", origin="lower")

shifted = np.interp(shifted, wave[0], np.arange(len(wave[0])))
plt.plot(shifted, np.arange(len(shifted)), "k--")
plt.show()
