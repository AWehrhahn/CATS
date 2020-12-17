import logging
from os.path import dirname, join

import numpy as np
import pandas as pd

from astropy import units as u
from astropy.utils.iers import IERS_Auto

from cats.simulator.detector import Crires
from cats.extractor.runner import CatsRunner

# TODO List:
# - automatically mask points before fitting with SME
# - if star and planet steps aren't run manually, we use the initial values
#   instead we should load the data if possible
# - Tests for all the steps
# - Refactoring of the steps, a lot of the code is strewm all over the place
# - Determine Uncertainties for each point

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
star = "HD209458"
planet = "b"

# Initialize the CATS runner
base_dir = dirname(__file__)
raw_dir = join(base_dir, "HD209458_v4")
runner = CatsRunner(
    detector, star, planet, linelist, base_dir=base_dir, raw_dir=raw_dir
)

# Override data with known information
star = runner.run_module("star", load=True)
runner.star.vsini = 1.2 * (u.km / u.s)
runner.star.monh = 0 * u.one
runner.star.name = "HD209458"
runner.star.radial_velocity = -14.743 * (u.km / u.s)

planet = runner.run_module("planet", load=True)
runner.planet.inc = 86.59 * u.deg
runner.planet.ecc = 0 * u.one
runner.planet.period = 3.52472 * u.day

# Run the Runner
data = runner.run(["solve_problem"])
pass

# TODO:
# Adjust the mask manually
