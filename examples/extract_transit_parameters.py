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
from cats.spectrum import SpectrumArray
from cats.extractor.extract_transit_parameters import extract_transit_parameters
from exoorbit.bodies import Planet, Star
from exoorbit.orbit import Orbit


if __name__ == "__main__":
    data_dir = join(dirname(__file__), "noise_1", "raw")
    target_dir = join(dirname(__file__), "noise_1", "medium")
    files = join(data_dir, "*.fits")

    detector = Crires("H/1/4", [1, 2, 3])
    star = Star.load(join(target_dir, "star.yaml"))
    spectra = SpectrumArray.read(join(target_dir, "spectra_normalized.npz"))

    planet = extract_transit_parameters(spectra, star)

    fname = join(target_dir, "planet.yaml")
    planet.save(fname)
