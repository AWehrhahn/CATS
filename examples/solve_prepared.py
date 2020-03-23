import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.time import Time
from astropy.constants import c
from astropy import units as u

from cats.simulator.detector import Crires
from cats.data_modules.stellar_db import StellarDb
from cats.reference_frame import PlanetFrame, TelescopeFrame


from cats.solver.solution import __difference_matrix__, best_lambda, Tikhonov
from cats.solver.linear import LinearSolver

detector = Crires("H/1/4", [1, 2, 3])
sdb = StellarDb()
star = sdb.get("HD209458")
planet = star.planets["b"]

transit_time = "2020-05-25T10:31:25.418"
transit_time = Time(transit_time, format="fits")
planet.time_of_transit = transit_time

wavelength = np.load("wave.npy")
times = np.load("times.npy")
img_spectra = np.load("spectra.npy")
img_telluric = np.load("telluric.npy")
img_stellar = np.load("stellar.npy")
img_intensities = np.load("intensities.npy")
planet_model = np.load("planet_model.npy")
times = Time(times, format="fits")

solver = LinearSolver(detector, star, planet)
wave, x0 = solver.solve(
    times,
    wavelength,
    img_spectra,
    img_stellar,
    img_intensities,
    img_telluric,
    regweight=200,
)

np.save("planet_spectrum.npy", x0)
np.save("wavelength_planet.npy", wave)

plt.plot(wavelength[32], planet_model)
plt.plot(wave, x0)
plt.savefig("planet_spectrum.png")
