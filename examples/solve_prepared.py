from os.path import dirname, join

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.constants import c
from astropy.time import Time
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from tqdm import tqdm

from cats.data_modules.stellar_db import StellarDb
from cats.reference_frame import PlanetFrame, TelescopeFrame
from cats.simulator.detector import Crires
from cats.solver.linear import LinearSolver
from cats.spectrum import SpectrumArray

from exoorbit.bodies import Star, Planet

if __name__ == "__main__":
    medium_dir = join(dirname(__file__), "noise_1", "medium")
    done_dir = join(dirname(__file__), "noise_1", "done")

    detector = Crires("H/1/4", [1, 2, 3])
    star = Star.load(join(medium_dir, "star.yaml"))
    planet = Planet.load(join(medium_dir, "planet.yaml"))

    transit_time = "2020-05-25T10:31:25.418"
    transit_time = Time(transit_time, format="fits")
    planet.time_of_transit = transit_time

    print("Loading data...")
    spectra = SpectrumArray.read(join(medium_dir, "spectra_normalized.npz"))
    telluric = SpectrumArray.read(join(medium_dir, "telluric.npz"))
    stellar = SpectrumArray.read(join(medium_dir, "stellar.npz"))
    intensities = SpectrumArray.read(join(medium_dir, "intensities.npz"))

    # regweight:
    # for noise 0:  200
    # for noise 1%: 2000
    print("Solving the problem...")
    times = spectra.datetime
    wavelength = spectra.wavelength.to_value(u.AA)
    spectra = spectra.flux.to_value(1)
    telluric = telluric.flux.to_value(1)
    stellar = stellar.flux.to_value(1)
    intensities = intensities.flux.to_value(1)

    solver = LinearSolver(detector, star, planet)
    wave, x0 = solver.solve(
        times, wavelength, spectra, stellar, intensities, telluric, regweight=1,
    )

    print("Saving data...")
    np.save(join(done_dir, "planet_spectrum_noise_1.npy"), x0)
    np.save(join(done_dir, "wavelength_planet_noise_1.npy"), wave)

    print("Plotting results...")
    planet_model = np.load(join(medium_dir, "planet_model.npy"))

    plt.plot(wave, x0)
    plt.plot(wavelength[32], planet_model)
    plt.show()
    plt.savefig(join(done_dir, "planet_spectrum_noise_1.png"))
