from os.path import dirname, join

import batman
import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import c
from astropy.time import Time
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from tqdm import tqdm

from cats.data_modules.stellar_db import StellarDb
from cats.simulator.detector import Crires
from exoorbit.bodies import Planet, Star
from exoorbit.orbit import Orbit


def log_likelihood(theta):
    params.t0 = theta[0]
    try:
        model = m.light_curve(params)
        chisq = -0.5 * np.nansum((y - model) ** 2)
        if np.isfinite(chisq):
            return chisq
    except:
        pass
    finally:
        return -np.inf


def extract_transit_parameters(spectra, star, planet):
    transit_time = "2020-05-25T10:31:25.418"
    transit_time = Time(transit_time, format="fits")
    planet.transit_time = transit_time

    # TODO actually fit the transit
    print("TODO: Actually fit the planet transit in the data!")
    return planet

    wave = np.load("wave.npy")
    times = np.load("times.npy")
    img_spectra = np.load("spectra.npy")
    img_telluric = np.load("telluric.npy")
    img_stellar = np.load("stellar.npy")

    times = Time(times, format="fits")

    simulation = img_stellar * img_telluric
    simulation = np.nan_to_num(simulation)
    simulation = gaussian_filter1d(simulation, detector.spectral_broadening, axis=1)

    # TODO Fit planet transit orbit to this curve
    # Can't seem to find the transit?
    ys = np.nansum(simulation, axis=1)
    y = np.nanmedian(img_spectra / simulation, axis=1)

    plt.plot(times.mjd, y, label="observation")
    plt.show()

    # Setup batman model
    params = batman.TransitParams()  # object to store transit parameters
    params.t0 = planet.t0.mjd  # time of inferior conjunction
    params.per = planet.period.to_value("day")  # orbital period
    params.rp = (planet.radius / star.radius).to_value(
        1
    )  # planet radius (in units of stellar radii)
    params.a = (planet.sma / star.radius).to_value(
        1
    )  # semi-major axis (in units of stellar radii)
    params.inc = planet.inc.to_value("deg")  # orbital inclination (in degrees)
    params.ecc = planet.ecc.to_value(1)  # eccentricity
    params.w = planet.w.to_value("deg")  # longitude of periastron (in degrees)
    params.limb_dark = "nonlinear"  # limb darkening model
    params.u = [0.5, 0.1, 0.1, -0.1]  # limb darkening coefficients [u1, u2, u3, u4]

    t = times.mjd  # times at which to calculate light curve
    m = batman.TransitModel(params, t)  # initializes model

    pos = planet.t0.mjd + 1e-4 * np.random.randn(32, 1)
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood)
    sampler.run_mcmc(pos, 5000, progress=True)

    flat_samples = sampler.get_chain(discard=100, flat=True)
    fig = corner.corner(flat_samples)
    plt.show()

    t0 = np.median(flat_samples)
    params.t0 = t0

    model = m.light_curve(params)

    plt.plot(times.mjd, y, label="observation")
    plt.plot(times.mjd, model, label="model")
    plt.show()

    # # TODO: plot the expected position of a spectral line on top
    # rv = orbit.radial_velocity_planet(times)
    # central_wavelength = 15000

    # beta = (rv / c).to_value(1)
    # shifted = np.sqrt((1 + beta) / (1 - beta)) * central_wavelength

    # plt.imshow(img_spectra, aspect="auto", origin="lower", vmin=0.98, vmax=1.01)

    # shifted = np.interp(shifted, wave[0], np.arange(len(wave[0])))
    # plt.plot(shifted, np.arange(len(shifted)), "k--")
    # plt.show()


if __name__ == "__main__":
    data_dir = join(dirname(__file__), "noise_1", "raw")
    target_dir = join(dirname(__file__), "noise_1", "medium")
    files = join(data_dir, "*.fits")

    detector = Crires("H/1/4", [1, 2, 3])
    sdb = StellarDb()
    star = sdb.get("HD209458")
    planet = star.planets["b"]
    star = Star.load(join(target_dir, "star.yaml"))

    planet = extract_transit_parameters(spectra, star, planet)

    fname = join(target_dir, "planet.yaml")
    planet.save(fname)
