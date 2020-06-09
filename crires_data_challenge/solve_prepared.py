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
from cats.solver.spline import SplineSolver
from cats.solver.bayes import BayesSolver
from cats.spectrum import SpectrumArray, SpectrumList

from exoorbit.bodies import Star, Planet


def solve_prepared(
    spectra,
    telluric,
    stellar,
    intensities,
    detector,
    star,
    planet,
    seg=5,
    solver="linear",
):
    # regweight:
    # for noise 0:  1
    # for noise 1%: 23
    print("Solving the problem...")
    spectra = spectra.get_segment(seg)
    telluric = telluric.get_segment(seg)
    stellar = stellar.get_segment(seg)
    intensities = intensities.get_segment(seg)

    times = spectra.datetime
    wavelength = spectra.wavelength.to_value(u.AA)
    spectra = spectra.flux.to_value(1)
    telluric = telluric.flux.to_value(1)
    stellar = stellar.flux.to_value(1)
    intensities = intensities.flux.to_value(1)

    if solver == "linear":
        solver = LinearSolver(
            detector,
            star,
            planet,
            regularization_ratio=1,
            plot=True,
            regularization_weight=1,
        )
    elif solver == "spline":
        solver = SplineSolver(detector, star, planet)
    elif solver == "bayesian":
        solver = BayesSolver(detector, star, planet)
    else:
        raise ValueError(
            "Unrecognized solver option {solver} expected one of ['linear', 'spline', 'bayesian']"
        )
    spec = solver.solve(times, wavelength, spectra, stellar, intensities, telluric)

    return spec


if __name__ == "__main__":
    medium_dir = join(dirname(__file__), "medium")
    done_dir = join(dirname(__file__), "done")

    detector = Crires("K/2/4", [1, 2, 3], orders=[2, 3, 4, 5, 6, 7])
    star = Star.load(join(medium_dir, "star.yaml"))
    planet = Planet.load(join(medium_dir, "planet.yaml"))

    transit_time = "2020-05-25T10:31:25.418"
    transit_time = Time(transit_time, format="fits")
    planet.time_of_transit = transit_time

    print("Loading data...")
    normalized = SpectrumArray.read(join(medium_dir, "spectra_normalized.npz"))
    telluric = SpectrumArray.read(join(medium_dir, "telluric.npz"))
    stellar = SpectrumArray.read(join(medium_dir, "stellar.npz"))
    intensities = SpectrumArray.read(join(medium_dir, "intensities.npz"))

    spec = solve_prepared(
        normalized, telluric, stellar, intensities, detector, star, planet
    )

    print("Saving data...")
    spec.write("planet_noise_1.fits")

    print("Plotting results...")
    planet_model = SpectrumList.read(join(done_dir, "planet_model.fits"))

    plt.plot(spec.wavelength, spec.flux)
    plt.plot(
        np.concatenate(planet_model.wavelength),
        gaussian_filter1d(np.concatenate(planet_model.flux), 1),
    )
    plt.show()
    plt.savefig(join(done_dir, "planet_spectrum_noise_1.png"))
