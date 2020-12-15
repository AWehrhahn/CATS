import logging
from os.path import dirname, join

import numpy as np
import pandas as pd

from astropy import units as u
from astropy.utils.iers import IERS_Auto

from cats.simulator.detector import Crires

from runner import CatsRunner

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

raw_dir = join(dirname(__file__), "HD209458_v4")
runner = CatsRunner(detector, star, planet, linelist, raw_dir=raw_dir)

# Override data with known information
runner.star.vsini = 1.2 * (u.km / u.s)
runner.star.monh = 0 * u.one
runner.star.name = "HD209458"
runner.star.radial_velocity = -14.743 * (u.km / u.s)

runner.planet.inc = 86.59 * u.deg
runner.planet.ecc = 0 * u.one
runner.planet.period = 3.52472 * u.day

# Run the Runner
data = runner.run(["solve_problem"])
pass

# TODO:
# Adjust the mask manually
