from os.path import dirname, join, exists

import numpy as np
import matplotlib.pyplot as plt

from cats.simulator.detector import Crires
from cats.spectrum import SpectrumList, SpectrumArray
from cats.data_modules.stellar_db import StellarDb

from cats.extractor.extract_stellar_parameters import extract_stellar_parameters
from cats.extractor.extract_transit_parameters import extract_transit_parameters
from cats.extractor.normalize_observation import normalize_observation
from cats.extractor.prepare import create_intensities
from cats.extractor.prepare import create_stellar
from cats.extractor.prepare import create_telluric

from collect_observations import collect_observations
from solve_prepared import solve_prepared
from simulate_planet import simulate_planet

from astroplan import download_IERS_A

# download_IERS_A()

if __name__ == "__main__":
    base_dir = join(dirname(__file__), "noise_5")
    raw_dir = join(base_dir, "raw")
    medium_dir = join(base_dir, "medium")
    done_dir = join(base_dir, "medium")

    detector = Crires("H/1/4", [1, 2, 3])
    wrange = detector.regions
    observatory = detector.observatory
    linelist = join(base_dir, "crires_h_1_4.lin")

    sdb = StellarDb()
    star = sdb.get("HD209458")

    fname = join(medium_dir, "spectra.npz")
    if not exists(fname):
        files = join(raw_dir, "*.fits")
        spectra = collect_observations(files)
        spectra.write(fname)
    else:
        spectra = SpectrumArray.read(fname)
    times = spectra.datetime
    # TODO: determine tellurics
    telluric = create_telluric(wrange, spectra, star, observatory, times)
    telluric.write(join(medium_dir, "telluric.npz"))

    star = extract_stellar_parameters(spectra, star, detector.blaze, linelist)
    star.save(join(medium_dir, "star.yaml"))

    stellar = create_stellar(wrange, spectra, star, times, linelist)
    stellar.write(join(medium_dir, "stellar.npz"))

    spectra = normalize_observation(spectra, stellar, telluric, detector)
    spectra.write(join(medium_dir, "spectra_normalized.npz"))

    # TODO: proper fit of transit parameters
    planet = extract_transit_parameters(spectra, star)
    planet.save(join(medium_dir, "planet.yaml"))

    intensities = create_intensities(
        wrange, spectra, star, planet, observatory, times, linelist
    )
    intensities.write(join(medium_dir, "intensities.npz"))

    spec = solve_prepared(
        spectra, telluric, stellar, intensities, detector, star, planet
    )

    print("Saving data...")
    spec.write(join(done_dir, "planet_extracted.fits"))

    print("Plotting results...")
    planet_model = simulate_planet(wavelength, star, planet, detector)
    planet_model("planet_model.fits")

    plt.plot(spec.wavelength, spec.flux)
    plt.plot(planet_model.wavelength, planet_model.flux)
    plt.show()
    plt.savefig(join(done_dir, "planet_spectrum.png"))
