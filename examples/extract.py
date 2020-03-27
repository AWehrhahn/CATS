from os.path import dirname, join

import numpy as np
import matplotlib.pyplot as plt

from cats.simulator.detector import Crires
from cats.spectrum import SpectrumList
from cats.data_modules.stellar_db import StellarDb

from cats.extractor.collect_observations import collect_observations
from cats.extractor.extract_stellar_parameters import extract_stellar_parameters
from cats.extractor.extract_transit_parameters import extract_transit_parameters
from cats.extractor.normalize_observation import normalize_observation
from cats.extractor.prepare import create_intensities, create_stellar, create_telluric
from solve_prepared import solve_prepared

from astroplan import download_IERS_A, IERS_A_in_cache

if not IERS_A_in_cache():
    download_IERS_A()

if __name__ == "__main__":
    base_dir = join(dirname(__file__), "noise_1")
    raw_dir = join(base_dir, "raw")
    medium_dir = join(base_dir, "medium")
    done_dir = join(base_dir, "medium")

    files = join(raw_dir, "*.fits")
    linelist = join(base_dir, "crires_h_1_4.lin")
    detector = Crires("H/1/4", [1, 2, 3])
    wrange = detector.regions
    observatory = detector.observatory

    sdb = StellarDb()
    star = sdb.get("HD209458")

    spectra = collect_observations(files)
    times = spectra.datetime
    # TODO: determine tellurics
    telluric = create_telluric(wrange, spectra, star, observatory, times)
    star = extract_stellar_parameters(spectra, star, detector.blaze, linelist)
    stellar = create_stellar(wrange, spectra, star, times)
    spectra = normalize_observation(spectra, stellar, telluric, detector)
    # TODO: proper fit of transit parameters
    planet = extract_transit_parameters(spectra, star)
    intensities = create_intensities(wrange, spectra, star, planet, observatory, times)
    wave, x0 = solve_prepared(
        spectra, telluric, stellar, intensities, detector, star, planet
    )

    print("Saving data...")
    np.save(join(done_dir, "planet_spectrum_noise_1.npy"), x0)
    np.save(join(done_dir, "wavelength_planet_noise_1.npy"), wave)

    print("Plotting results...")
    planet_model = SpectrumList.read(join(medium_dir, "planet_model.fits"))

    plt.plot(wave, x0)
    plt.plot(planet_model.wavelength, planet_model.flux)
    plt.show()
    plt.savefig(join(done_dir, "planet_spectrum_noise_1.png"))
