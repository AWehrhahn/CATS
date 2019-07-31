"""
CATS - Characterization of exoplanet Atmospheres with Transit Spectroscopy

author: Ansgar Wehrhahn
"""
import logging
import importlib

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import least_squares

from . import config, solution
from .data_modules.dataset import dataset
from .orbit import Orbit as orbit_calculator

func_mapping = {
    "parameters": "get_parameters",
    "observations": "get_observations",
    "stellar_flux": "get_stellarflux",
    "intensities": "get_intensities",
    "tellurics": "get_tellurics",
    "planet": "get_planet"
}


def load_configuration(star, planet, fname="config.yaml"):
    # Step 1: Load values from yaml/json file
    logging.info("Load configuration")
    configuration = config.load_config(filename=fname, star=star, planet=planet)
    return configuration

def load_module(name, configuration):
    # Load modules with the given name
    name = name.lower()
    mod_name = f".data_modules.{name}"
    lib = importlib.import_module(mod_name, package="cats")
    module = getattr(lib, name)(configuration)
    return module

def load_data(star, planet, configuration):
    logging.info("Load data from modules")
    # Step 0: Get modules for each step from configuration
    steps = configuration["workflow"]
    modules = {s: configuration[s] for s in steps}
    for k, m in modules.items():
        conf = configuration[m] if m in configuration.keys() else {}
        modules[k] = load_module(m, conf)

    # Step 1: Load data
    data = {}
    for s in steps:
        logging.info("Loading %s data from module %s", s, str(modules[s]))
        data[s] = getattr(modules[s], func_mapping[s])(**data)
        modules[s]._data_from_other_modules = data
    return data


def calculate_solution(data, configuration, lamb="auto"):
    parameters = data["parameters"]
    tell = data["tellurics"]
    i_atm = data["intensities"][0]
    i_planet = data["intensities"][1]
    flux = data["stellar_flux"]
    obs = data["observations"]

    area_planet = parameters["A_planet"].value
    area_atmosphere = parameters['A_atm'].value

    orbit = orbit_calculator(configuration, parameters)

    plot_observation_timeseries(obs, parameters)

    # Step 1: Shift data onto the same wavelength grid (rv = 0)
    # rv of observations is given by system velocity + orbit parameters (at the observation time)
    # Everything else is already in barycentric (?)
    logging.info("Shift observations into rest frame")
    for i in range(len(obs.data)):
        obs.shift(orbit.get_rv(obs.time[i]), i)

    # plot_observation_timeseries(obs)
    # Step 2: Broaden theoretical spectra with instrumental profile
    # TODO Only apply to theoretical spectra, not measured
    logging.info("Applying instrumental broadening to theoretical spectra")
    broadening = configuration["broadening"]
    # tell.broadening = broadening
    # i_atm.broadening = broadening
    # i_planet.broadening = broadening
    flux.broadening = broadening

    # Step 3: Calculate intermediate values f and g
    logging.info("Calculating factors f and g")
    f = i_atm * tell
    g = obs - (flux - i_planet * area_planet) * tell

    # Collapse g
    wave = g.wave
    g = np.sum(g.data, axis=0) / len(g.data)

    if isinstance(f, dataset):
        f.new_grid(wave)
        f = np.sum(f.data, axis=0) / len(f.data)
    else:
        f = np.full(len(g), f)

    # Step 4 (optional): Determine regularization parameter
    if lamb == "auto":
        logging.info("Determining best regulariztion parameter lambda")
        lamb = solution.best_lambda(f, g, plot=True)
        logging.info("Best lambda value: %.3e", lamb)
    # Step 5: Solve linear equation system
    logging.info("Solve inverse problem")
    result = solution.Tikhonov(f, g, lamb)

    # Step 6: Normalize
    # TODO: Is that ok?
    a_atm = result.max() - result.min()
    result -= result.min()
    result /= np.percentile(result, 95)

    result = dataset(wave, result)
    return result

def plot_observation_timeseries(obs, parameters):
    nobs = len(obs.data)
    nwave = len(obs.wave)
    wave = obs.wave

    #sort by phase
    times = obs.phase
    sort = np.argsort(times)
    datacube = obs.data[sort]
    times = times[sort]

    # if False:
    white = np.median(datacube, axis=1)
    stellar = np.median(datacube, axis=0)
    # datacube /= white[:, None]
    datacube /= stellar

    plt.subplot(211)
    plt.plot(times, white)

    plt.subplot(212)
    lower, upper = np.nanpercentile(datacube, (5, 95))
    plt.imshow(datacube, origin="lower", aspect="auto", cmap="gray", vmin=lower, vmax=upper)
    plt.xticks(np.arange(nwave)[::100], wave[::100])
    plt.yticks(np.arange(nobs)[::10], times[::10])

    orbit = orbit_calculator(None, parameters)
    lower, upper = orbit.maximum_phase()
    mid = (lower + upper) / 2

    lower, mid, upper = np.digitize((lower, mid, upper), times)

    plt.hlines([lower, upper], 0, nwave, color="r", linestyles="dashed")
    plt.hlines(mid, 0, nwave, color="r")
    plt.show()

def plot(results, data, configuration):
    flux = data["stellar_flux"]
    tell = data["tellurics"]
    if "planet" in data.keys():
        planet = data["planet"]
    else:
        planet = None


    plt.plot(results.wave, results.data, label="planet recovered")
    plt.plot(flux.wave, flux.data, label="stellar_flux")
    plt.plot(tell.wave, tell.data, label="tellurics")
    if planet is not None:
        plt.plot(planet.wave, planet.data, label="planet input")
    plt.legend(loc="best")

    plt.xlim(results.wave[0], results.wave[-1])
    plt.ylim(0, 1.05)

    plt.show()

def main(star, planet, lamb="auto"):
    logging.debug("------------")
    # Step 1: Load configuration
    configuration = load_configuration(star, planet)
    # Step 2: Load data from modules
    data = load_data(star, planet, configuration)
    # Step 3: Calculate solution
    results = calculate_solution(data, configuration, lamb=lamb)
    # Step 4: Plot results
    plot(results, data, configuration)
    logging.debug("------------")
    return results
