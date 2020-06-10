from glob import glob
from os.path import basename, dirname, exists, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.io import fits
from astropy.time import Time
from exoorbit.bodies import Planet, Star
from tqdm import tqdm

from cats.data_modules.stellar_db import StellarDb
from cats.extractor.extract_stellar_parameters import (
    extract_stellar_parameters,
    first_guess,
    fit_observation,
)
from cats.extractor.extract_transit_parameters import extract_transit_parameters
from cats.extractor.normalize_observation import normalize_observation
from cats.extractor.prepare import create_intensities, create_stellar, create_telluric
from cats.simulator.detector import Crires
from cats.spectrum import Spectrum1D, SpectrumArray, SpectrumList
from pysme.sme import SME_Structure

from simulate_planet import simulate_planet
from solve_prepared import solve_prepared
from hitran_linelist import Hitran

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
        orders = list(range(wave.shape[1]))
        for order in orders[::-1]:
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
orders = [7, 6, 5, 4, 3, 2]
# orders = [2, 3, 4, 5, 6, 7]
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
done_dir = join(dirname(__file__), "done")

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
if not exists(fname) or False:
    telluric = create_telluric(wrange, spectra, star, observatory, times)
    telluric.write(fname)
else:
    telluric = SpectrumArray.read(fname)

# 4: Extract stellar parameters
fname = join(medium_dir, "first_guess.sme")
if not exists(fname) or False:
    sme = first_guess(spectra, star, detector.blaze, linelist, detector)
    sme.save(fname)
else:
    sme = SME_Structure.load(fname)
    sme.vrad_flag = "fix"
    sme.cscale_flag = "fix"

# TODO:
# Adjust the mask manually

fname = join(medium_dir, "star.yaml")
if not exists(fname) or False:
    sme, star = fit_observation(sme, star)
    star.save(fname)
else:
    star = Star.load(fname)

# 5: Create stellar spectra
fname = join(medium_dir, "stellar_combined.npz")
if not exists(fname) or False:
    combined = create_stellar(
        wrange,
        spectra,
        times,
        method="combine",
        blaze=detector.blaze,
        mask=sme.mask_cont,
        telluric=telluric,
        detector=detector,
    )
    combined.write(fname)
else:
    combined = SpectrumArray.read(fname)

fname = join(medium_dir, "stellar.npz")
if not exists(fname) or False:
    stellar = create_stellar(
        wrange, spectra, times, method="sme", star=star, linelist=linelist
    )
    stellar.write(fname)
else:
    stellar = SpectrumArray.read(fname)


# 6: Normalize observations
fname = join(medium_dir, "spectra_normalized.npz")
if not exists(fname) or False:
    normalized = normalize_observation(spectra, stellar, telluric, detector)
    normalized.write(fname)
else:
    normalized = SpectrumArray.read(fname)

# TODO: Should these be the same?
# sort = np.argsort(normalized.datetime)
# # The telluric absorption influences the transmision spectrum
# norm = np.nanmean(telluric.flux[sort], axis=1)
# plt.plot(np.nanmean(spectra.flux[sort] / norm[:, None] / 3e4, axis=1))
# plt.plot(np.nanmean(normalized.flux[sort] / norm[:, None], axis=1))
# plt.show()

# 7: Determine Planet transit
# TODO: proper fit of transit parameters
fname = join(medium_dir, "planet.yaml")
if not exists(fname) or False:
    planet = extract_transit_parameters(spectra, telluric, star, planet)
    planet.save(fname)
else:
    planet = Planet.load(fname)


# 8: Create specific intensitiies
fname = join(medium_dir, "intensities.npz")
if not exists(fname) or False:
    intensities = create_intensities(
        wrange, normalized, star, planet, observatory, times, linelist
    )
    intensities.write(fname)
else:
    intensities = SpectrumArray.read(fname)

i = 51
j = 5
plt.plot(intensities[i].wavelength[j], intensities[i].flux[j])
plt.show()

# We use sme to estimate the limb darkening at each point
# But use the extracted stellar spectrum, to create the actual intensities
intensities /= stellar
intensities *= combined

plt.plot(intensities[i].wavelength[j], intensities[i].flux[j])
plt.show()


# 9: Solve the equation system
wavelength = normalized[0].wavelength
flux = normalized[0].flux

hitran = Hitran()

for segment in tqdm(range(18)):
    spec = solve_prepared(
        normalized,
        telluric,
        combined,
        intensities,
        detector,
        star,
        planet,
        seg=segment,
        solver="linear",
    )

    print("Saving data...")
    spec.write(join(done_dir, f"planet_extracted_{segment}.fits"))

    # print("Plotting results...")
    # planet_model = simulate_planet(spectra[0].wavelength, star, planet, detector)
    # planet_model.write("planet_model.fits")

    plt.plot(
        wavelength[segment], flux[segment], label="normalized observation",
    )
    plt.plot(spec.wavelength, spec.flux, label="extracted")
    # plt.vlines(hitran.wavelength, 0, 2)
    plt.xlim(wavelength[segment][0].value, wavelength[segment][-1].value)

    # plt.plot(planet_model.wavelength, planet_model.flux)
    plt.ylabel("Flux, normalised")
    plt.xlabel("Wavelength [Å]")
    plt.legend()
    # plt.show()
    plt.savefig(join(done_dir, f"planet_spectrum_{segment}.png"))
    plt.clf()
