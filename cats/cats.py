"""
CATS - Characterization of exoplanet Atmospheres with Transit Spectroscopy

author: Ansgar Wehrhahn
"""
import logging
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import least_squares

from . import config, solution
from .orbit import orbit as orbit_calculator

from .data_modules.space import space
from .data_modules.gj1214b import gj1214b
from .data_modules.nirspec import nirspec
from .data_modules.marcs import marcs
from .data_modules.aronson import aronson
from .data_modules.dataset import dataset


steps = ["parameters", "observations", "stellar_flux", "intensities", "tellurics"]
func_mapping = {
    "parameters": "get_parameters",
    "observations": "get_observations",
    "stellar_flux": "get_stellarflux",
    "intensities": "get_intensities",
    "tellurics": "get_tellurics",
}
modules_mapping = {"space": space, "gj1214b": gj1214b, "nirspec": nirspec, "marcs": marcs, "aronson":aronson}


def load_configuration(star, planet, fname="config.yaml"):
    # Step 1: Load values from yaml/json file
    logging.info("Load configuration")
    configuration = config.load_config(fname, star=star, planet=planet)
    return configuration


def load_data(star, planet, configuration):
    logging.info("Load data from modules")
    # Step 0: Get modules for each step from configuration
    modules = {s: configuration[s] for s in steps}
    for k, m in modules.items():
        modules[k] = modules_mapping[m](configuration)

    # Step 1: Load data
    data = {}
    for s in steps:
        logging.info("Loading %s data from module %s", s, str(modules[s]))
        data[s] = getattr(modules[s], func_mapping[s])(**data)
    return data


def calculate_solution(data, configuration, lamb="auto"):
    parameters = data["parameters"]
    tell = data["tellurics"]
    i_atm = data["intensities"][0]
    i_planet = data["intensities"][1]
    flux = data["stellar_flux"]
    obs = data["observations"]

    area_planet = parameters["A_planet"]
    area_atmosphere = parameters['A_atm']

    orbit = orbit_calculator(configuration, parameters)

    plot_observation_timeseries(obs, parameters)


    plot_observation_timeseries(obs, parameters)


    # Step 1: Shift data onto the same wavelength grid (rv = 0)
    # rv of observations is given by system velocity + orbit parameters (at the observation time)
    # Everything else is already in barycentric (?)
    logging.info("Shift observations into rest frame")
    for i in range(len(obs.data)):
        obs.shift(i, orbit.get_rv(obs.time[i]))

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
    f = tell * i_atm * area_atmosphere
    g = obs - (flux - i_planet * area_planet) * tell
    
    # Collapse g
    wave = g.wave
    g = np.sum(g.data, axis=0) / len(g.data)
    
    if isinstance(f, dataset):
        f._interpolate(wave)
        f = f.data
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

    result = dataset(wave, result)
    return result

def plot_observation_timeseries(obs, parameters):
    nobs = len(obs.data)
    nwave = len(obs.wave)
    wave = obs.wave

    #sort by time
    sort = np.argsort(obs.time)
    datacube = obs.data[sort]
    dates = obs.time[sort]

    plt.imshow(datacube, origin="lower", aspect="auto", cmap="gray")
    plt.xticks(np.arange(nwave)[::100], wave[::100])
    plt.yticks(np.arange(nobs)[::10], dates[::10])
    plt.show()

def plot(results, data, configuration):
    plt.plot(results.wave, results.data)
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
