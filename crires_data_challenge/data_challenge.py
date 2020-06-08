from os.path import dirname, join, exists, basename
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
from astropy import units as u
from astropy.time import Time
from tqdm import tqdm

from cats.simulator.detector import Crires
from cats.spectrum import SpectrumList, SpectrumArray, Spectrum1D
from cats.data_modules.stellar_db import StellarDb

from cats.extractor.extract_stellar_parameters import extract_stellar_parameters
from cats.extractor.extract_transit_parameters import extract_transit_parameters
from cats.extractor.normalize_observation import normalize_observation
from cats.extractor.prepare import create_intensities
from cats.extractor.prepare import create_stellar
from cats.extractor.prepare import create_telluric


from exoorbit.bodies import Star, Planet

# from solve_prepared import solve_prepared
# from simulate_planet import simulate_planet

# from astroplan import download_IERS_A
# download_IERS_A()


def collect_observations(files, additional_data):
    files = glob(files)
    additional_data = pd.read_csv(additional_data)

    speclist = []
    for f in tqdm(files):
        i = int(basename(f)[9:-5])
        hdu = fits.open(f)
        wave = hdu[1].data << u.AA
        flux = hdu[2].data << u.one

        add = additional_data.iloc[i]
        time = Time(add["time"], format="jd")
        airmass = add["airmass"]
        rv = add["system velocity"] << (u.km / u.s)

        spectra = []
        for order in range(wave.shape[1]):
            for det in [1, 2, 3]:
                w = wave[det - 1, order]
                f = flux[det - 1, order]
                if np.all(np.isnan(w)) or np.all(np.isnan(f)):
                    continue

                spec = Spectrum1D(
                    flux=f,
                    spectral_axis=w,
                    source="CRIRES+ Data Challenge 1",
                    star=star,
                    planet=planet,
                    observatory_location=observatory,
                    datetime=time,
                    reference_frame="telescope",
                    radial_velocity=rv,
                    airmass=airmass,
                )
                spectra += [spec]

        speclist += [SpectrumList.from_spectra(spectra)]

    times = [spec.datetime for spec in speclist]
    sort = np.argsort(times)
    spectra = [speclist[i] for i in sort]
    times = [times[i] for i in sort]

    spectra = SpectrumArray(speclist)
    return spectra


# Settings
setting = "K/2/4"
detectors = [1, 2, 3]
orders = [2, 3, 4, 5, 6, 7]
detector = Crires(setting, detectors, orders=orders)
wrange = detector.regions
observatory = detector.observatory

# Star info
sdb = StellarDb()
star = sdb.get("HD209458")
planet = star.planets["b"]

# Data locations
raw_dir = join(dirname(__file__), "HD209458")
medium_dir = join(dirname(__file__), "medium")
done_dir = join(dirname(__file__), "medium")

# Other data
linelist = join(raw_dir, "crires_k_2_4.lin")
additional_data = join(raw_dir, "HD209458_additional_data.csv")

# 1: Collect observations
fname = join(medium_dir, "spectra.npz")
if not exists(fname) or False:
    files = join(raw_dir, "*.fits")
    spectra = collect_observations(files, additional_data)
    spectra.write(fname)
else:
    spectra = SpectrumArray.read(fname)
times = spectra.datetime

# 2: Extract Telluric information
# TODO: determine tellurics

# 3: Create Tellurics
fname = join(medium_dir, "telluric.npz")
if not exists(fname) or True:
    telluric = create_telluric(wrange, spectra, star, observatory, times)
    telluric.write(fname)
else:
    telluric = SpectrumArray.read(fname)

# 4: Extract stellar parameters
fname = join(medium_dir, "star.yaml")
if not exists(fname) or True:
    star = extract_stellar_parameters(spectra, star, detector.blaze, linelist)
    star.save(fname)
else:
    star = Star.load(fname)

# 5: Create stellar spectra
fname = join(medium_dir, "stellar.npz")
if not exists(fname) or True:
    stellar = create_stellar(wrange, spectra, star, times, linelist)
    stellar.write(fname)
else:
    stellar = SpectrumArray.read(fname)

# 6: Normalize observations
fname = join(medium_dir, "spectra_normalized.npz")
if not exists(fname) or True:
    normalized = normalize_observation(spectra, stellar, telluric, detector)
    normalized.write(fname)
else:
    normalized = SpectrumArray(fname)

# 7: Determine Planet transit
# TODO: proper fit of transit parameters

# 8: Create specific intensitiies
fname = join(medium_dir, "intensities.npz")
if not exists(fname) or True:
    intensities = create_intensities(
        wrange, normalized, star, planet, observatory, times, linelist
    )
    intensities.write(fname)
else:
    intensities = SpectrumArray.read(fname)

# 9: Solve the equation system
spec = solve_prepared(
    normalized, telluric, stellar, intensities, detector, star, planet
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
