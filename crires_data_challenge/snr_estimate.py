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
star = "WASP-107"
planet = "b"

# Initialize the CATS runner
base_dir = join(dirname(__file__), "../datasets/WASP-107b_SNR100")
raw_dir = join(base_dir, "Spectrum_00")
medium_dir = join(base_dir, "medium")
done_dir = join(base_dir, "done")
runner = CatsRunner(
    detector,
    star,
    planet,
    linelist,
    base_dir=base_dir,
    raw_dir=raw_dir,
    medium_dir=medium_dir,
    done_dir=done_dir,
)

# Override data with known information
# star = runner.run_module("star", load=True)
# planet = runner.run_module("planet", load=True)

# Run the Runnert
# data = runner.run(["planet_radial_velocity"])
data = runner.run(["cross_correlation"])

runner.steps["cross_correlation"].plot(data["cross_correlation"])

pass
