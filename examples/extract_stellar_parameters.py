"""
Option A:
  - resample all spectra to a common wavelength grid
  - add them together
  - fit to that
Pros:
  - easier to normalize
  - radial velocity shift between observations is small (if from the same transit)
Cons:
  - Loose some precision due to resampling

Option B:
  - calculate SME spectra without sampling to wavelength grid
  - do least-squares to each spectrum individually and add them together
Pros:
  - more precise, since no interpolation
Cons:
  - difficult to normalize individual spectra
  - more effort

Lets do Option A!

"""
from glob import glob
from os.path import dirname, join

import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

import astroplan as ap
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.constants import c
from pysme.linelist.vald import ValdFile
from pysme.sme import SME_Structure
from pysme.solve import SME_Solver
from pysme.synthesize import synthesize_spectrum, Synthesizer
from pysme.gui import plot_plotly
from tqdm import tqdm

from scipy.optimize import curve_fit
from exoorbit import Orbit, Star

from cats.simulator.detector import Crires
from cats.data_modules.stellar_db import StellarDb
from cats.spectrum import SpectrumArray
from cats.extractor.extract_stellar_parameters import extract_stellar_parameters

if __name__ == "__main__":
    data_dir = join(dirname(__file__), "noise_1", "raw")
    target_dir = join(dirname(__file__), "noise_1", "medium")
    util_dir = join(dirname(__file__), "noise_1")
    files = join(data_dir, "*.fits")

    linelist = join(util_dir, "crires_h_1_4.lin")

    detector = Crires("H/1/4", [1, 2, 3])
    blaze = detector.blaze

    # Load the nominal values for an initial guess
    sdb = StellarDb()
    star = sdb.get("HD209458")

    # Load data
    print("Loading data...")
    spectra = SpectrumArray.read(join(target_dir, "spectra.npz"))
    times = spectra.datetime

    star = extract_stellar_parameters(spectra, star, blaze, linelist)

    fname = join(target_dir, "star.yaml")
    star.save(fname)
    pass
