import numpy as np
from astropy.time import Time
import matplotlib.pyplot as plt

from cats.data_modules.psg import PsgPlanetSpectrum
from cats.data_modules.stellar_db import StellarDb
from cats.simulator.detector import Crires
from cats.spectrum import SpectrumList

from os.path import join, dirname


def simulate_planet(wavelength, star, planet, detector):
    wrange = detector.regions

    psg = PsgPlanetSpectrum(star, planet)
    spec = psg.get(wrange, Time.now())
    spec = spec.resample(wavelength)
    spec = detector.apply_instrumental_broadening(spec)

    return spec


if __name__ == "__main__":
    detector = Crires("H/1/4", [1, 2, 3])
    wrange = detector.regions

    sdb = StellarDb()
    star = sdb.get("HD209458")
    planet = star.planets["b"]

    print("Loading data...")
    data_dir = join(dirname(__file__), "noise_1", "raw")
    files = join(data_dir, "HD209458_b_0.fits")
    data = SpectrumList.read(files)
    wave = data.wavelength

    print("Creating planet spectrum...")
    spec = simulate_planet(wave, star, planet, detector)

    print("Saving data...")
    spec.write("planet_model.fits")
