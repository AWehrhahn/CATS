"""
Normalize the observation
"""
from glob import glob
from os.path import dirname, join

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from scipy.optimize import curve_fit
from tqdm import tqdm

from cats.simulator.detector import Crires
from cats.spectrum import SpectrumArray, SpectrumList
from cats.extractor.normalize_observation import normalize_observation

if __name__ == "__main__":
    data_dir = join(dirname(__file__), "noise_1", "raw")
    target_dir = join(dirname(__file__), "noise_1", "medium")
    files = join(data_dir, "*.fits")

    linelist = f"{data_dir}/crires_h_1_4.lin"

    detector = Crires("H/1/4", [1, 2, 3])
    observatory = detector.observatory
    wrange = detector.regions

    print("Loading data...")
    spectra = SpectrumArray.read(join(target_dir, "spectra.npz"))
    stellar = SpectrumArray.read(join(target_dir, "stellar.npz"))
    telluric = SpectrumArray.read(join(target_dir, "telluric.npz"))

    print("Normalizing spectra...")
    spectra = normalize_observation(spectra, stellar, telluric, detector)

    print("Saving normalized data...")
    spectra = SpectrumArray(spectra)
    spectra.write(join(target_dir, "spectra_normalized.npz"))
