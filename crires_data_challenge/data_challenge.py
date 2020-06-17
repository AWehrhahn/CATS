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
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

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
from planet_spectra import load_planet

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
    speclist = [speclist[i] for i in sort]
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

# telluric.flux = gaussian_filter1d(telluric.flux, 1)

fname = join(medium_dir, "telluric_space.npz")
if not exists(fname) or False:
    telluric_space = create_telluric(
        wrange, spectra, star, observatory, times, source="space"
    )
    telluric_space.write(fname)
else:
    telluric_space = SpectrumArray.read(fname)

# 4: Extract stellar parameters
fname = join(medium_dir, "first_guess.sme")
if not exists(fname) or False:
    sme = first_guess(spectra, star, detector.blaze, linelist, detector)
    sme.save(fname)
else:
    sme = SME_Structure.load(fname)
    sme.vrad = sme.vrad.min()
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
fname = join(medium_dir, "stellar.npz")
if not exists(fname) or False:
    stellar = create_stellar(
        wrange, spectra, times, method="sme", star=star, linelist=linelist
    )
    stellar.write(fname)
else:
    stellar = SpectrumArray.read(fname)

fname = join(medium_dir, "stellar_combined.npz")
if not exists(fname) or False:
    combined = create_stellar(
        wrange,
        spectra,
        times,
        method="combine",
        mask=sme.mask_cont,
        telluric=telluric,
        detector=detector,
        stellar=stellar,
    )
    combined.write(fname)
else:
    combined = SpectrumArray.read(fname)


# 6: Normalize observations
fname = join(medium_dir, "spectra_normalized.npz")
if not exists(fname) or False:
    normalized = normalize_observation(spectra, stellar, telluric, detector)
    normalized.write(fname)
else:
    normalized = SpectrumArray.read(fname)

tmp = stellar.flux * telluric.flux
mask = np.isfinite(tmp)
func = (
    lambda s: gaussian_filter1d(telluric.flux[mask], s[1])
    * gaussian_filter1d(stellar.flux[mask], s[0])
    - normalized.flux[mask]
)

res = least_squares(func, x0=[1, 1], verbose=2)
detector.spectral_broadening = res.x[0].to_value(1)

telluric.flux = gaussian_filter1d(telluric.flux, res.x[1].to_value(1)) << u.one
stellar.flux = gaussian_filter1d(stellar.flux, res.x[0].to_value(1)) << u.one

# plt.plot(normalized.wavelength[0], normalized.flux[0])
# plt.plot(stellar.wavelength[0], tmp[0])
# plt.plot(stellar.wavelength[0], stellar.flux[0])
# plt.show()

# 7: Determine Planet transit
fname = join(medium_dir, "planet.yaml")
if not exists(fname) or False:
    planet = extract_transit_parameters(spectra, telluric, star, planet)
    planet.save(fname)
else:
    p = planet
    planet = Planet.load(fname)
    # planet.inc = p.inc
    # planet.ecc = p.ecc
    # planet.radius = p.radius


# 8: Create specific intensitiies
fname = join(medium_dir, "intensities.npz")
if not exists(fname) or False:
    intensities = create_intensities(
        wrange, normalized, star, planet, observatory, times, linelist
    )
    intensities.write(fname)
else:
    intensities = SpectrumArray.read(fname)

intensities.flux = gaussian_filter1d(intensities.flux, res.x[0].to_value(1)) << u.one

# TODO: use the ratio from SME and the spectrum from combined
# to create the intensities
sort = np.argsort(times)
i = np.arange(101)[sort][51]
plt.plot(normalized.wavelength[i], normalized.flux[i])
# plt.plot(combined.wavelength[i], combined.flux[i])
plt.plot(stellar.wavelength[i], stellar.flux[i])
plt.plot(intensities.wavelength[i], intensities.flux[i])
plt.plot(telluric.wavelength[i], telluric.flux[i])
plt.show()

# # We use sme to estimate the limb darkening at each point
# # But use the extracted stellar spectrum, to create the actual intensities
# intensities = (intensities / stellar) * combined

# plt.plot(intensities[i].wavelength[j], intensities[i].flux[j])
# plt.show()


# 9: Solve the equation system
wavelength = normalized[0].wavelength
flux = normalized[0].flux

planet_spectrum = load_planet()

# print("Shifting data to the telescope restframe")
# # TODO: add telluric
# for spec in [normalized, combined, intensities]:
#     spec.meta["star"] = star
#     spec.meta["planet"] = planet
#     spec.meta["observatory_location"] = observatory
#     spec.shift("telescope", inplace=True)


for segment in tqdm([15]):
    spec = solve_prepared(
        normalized,
        telluric,
        stellar,
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
    spec = spec.resample(wavelength[segment], "linear")
    ps = planet_spectrum.resample(wavelength[segment], "linear")

    plt.plot(
        wavelength[segment], flux[segment], label="normalized observation",
    )
    plt.plot(ps.wavelength, ps.flux, label="planet model")
    plt.plot(spec.wavelength, gaussian_filter1d(spec.flux, 10), label="extracted")
    # plt.vlines(hitran.wavelength, 0, 2)
    plt.xlim(wavelength[segment][0].value, wavelength[segment][-1].value)
    plt.ylim(0, 2)
    # plt.plot(planet_model.wavelength, planet_model.flux)
    plt.ylabel("Flux, normalised")
    plt.xlabel("Wavelength [Å]")
    plt.legend()
    # plt.show()
    plt.savefig(join(done_dir, f"planet_spectrum_{segment}.png"))
    plt.clf()

pass
