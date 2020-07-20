from glob import glob
from os.path import basename, dirname, exists, join
from copy import deepcopy

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

from cats.data_modules.telluric_fit import TelluricFit
from scipy.ndimage.morphology import binary_erosion, binary_dilation

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
        rv = add["barycentric velocity (Paranal)"] << (u.km / u.s)

        spectra = []
        orders = list(range(wave.shape[1]))
        for order in orders:
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


def normalize(spectra, stellar, telluric, detector):
    # Also broadening is matched to the observation
    # telluric and stellar have independant broadening factors
    sflux = stellar.flux
    tflux = telluric.flux

    for _ in range(3):
        normalized = normalize_observation(spectra, stellar, telluric, detector)

        tmp = sflux * tflux
        mask = np.isfinite(tmp)
        func = (
            lambda s: gaussian_filter1d(telluric.flux[mask], abs(s[1]))
            * gaussian_filter1d(stellar.flux[mask], abs(s[0]))
            - normalized.flux[mask]
        )
        res = least_squares(func, x0=[1, 1])
        stellar_broadening = abs(res.x[0].to_value(1))
        telluric_broadening = abs(res.x[1].to_value(1))

        detector.spectral_broadening = stellar_broadening
        tflux = gaussian_filter1d(telluric.flux, telluric_broadening) << u.one
        sflux = gaussian_filter1d(stellar.flux, stellar_broadening) << u.one

    return normalized, detector, stellar_broadening, telluric_broadening


def fit_tellurics(
    normalized, telluric, star, observatory, skip_resample=True, degree=1
):
    times = normalized.datetime
    t = TelluricFit(star, observatory, skip_resample=skip_resample, degree=degree)
    coeff = t.fit(normalized)
    airmass = t.calculate_airmass(times)
    model = t.model(coeff, airmass)

    mask = coeff[:, 0] > 0
    mask = ~binary_erosion(~mask)
    mask = ~binary_dilation(~mask, iterations=5)
    # Use morphology to improve mask
    for i in np.arange(coeff.shape[0])[mask]:
        coeff[i] = np.polyfit(airmass, telluric.flux[:, i], 1)

    model = t.model(coeff, t.calculate_airmass(times))
    model = model << u.one

    telluric = SpectrumArray(
        flux=model,
        spectral_axis=np.copy(spectra.wavelength),
        segments=spectra.segments,
        reference_frame="telescope",
        datetime=times,
        star=star,
        observatory_location=observatory,
    )
    return telluric


# TODO: barycentric velocity is given by the table, not from star

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
star.vsini = 1.2 * (u.km / u.s)
planet = star.planets["b"]

# Data locations
raw_dir = join(dirname(__file__), "HD209458_v4")
medium_dir = join(dirname(__file__), "medium_v4")
done_dir = join(dirname(__file__), "done_unnormalized")

# Other data
linelist = join(dirname(__file__), "crires_k_2_4.lin")
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
# Fit the observations against the airmass, using a second order polynomial?
# Assuming the water density is constant

# 3: Create Tellurics
fname = join(medium_dir, "telluric.npz")
if not exists(fname) or False:
    telluric = create_telluric(wrange, spectra, star, observatory, times)
    telluric.write(fname)
else:
    telluric = SpectrumArray.read(fname)

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
    # sme.vrad = sme.vrad.min()
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

# From info about the star
star.radial_velocity = -14.743 * (u.km / u.s)

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
fname = join(medium_dir, "normalized.npz")
normalized, detector, stellar_broadening, telluric_broadening = normalize(
    spectra, stellar, telluric, detector
)
normalized.write(fname)

detector.spectral_broadening = stellar_broadening
telluric.flux = gaussian_filter1d(telluric.flux, telluric_broadening) << u.one
stellar.flux = gaussian_filter1d(stellar.flux, stellar_broadening) << u.one


# 7: Determine Planet transit
fname = join(medium_dir, "planet.yaml")
if not exists(fname) or False:
    p = extract_transit_parameters(spectra, telluric, star, planet)
    p.save(fname)
else:
    p = Planet.load(fname)
    # This is based on what we know about the model
planet.t0 = p.t0
planet.inc = 86.59 * u.deg
planet.ecc = 0 * u.one
planet.period = 3.52472 * u.day
planet.radius = p.radius

# Fit Telluric Spectrum
telluric = fit_tellurics(
    normalized, telluric, star, observatory, skip_resample=True, degree=1
)

# 8: Create specific intensitiies
fname = join(medium_dir, "intensities.npz")
if not exists(fname) or False:
    intensities = create_intensities(
        wrange, normalized, star, planet, observatory, times, linelist
    )
    intensities.write(fname)
else:
    intensities = SpectrumArray.read(fname)

intensities.flux = gaussian_filter1d(intensities.flux, stellar_broadening) << u.one

# TODO: use the ratio from SME and the spectrum from combined
# to create the intensities
sort = np.argsort(times)
i = np.arange(101)[sort][51]
# plt.plot(normalized.wavelength[i], normalized.flux[i])
# # plt.plot(combined.wavelength[i], combined.flux[i])
# plt.plot(stellar.wavelength[i], stellar.flux[i])
# plt.plot(intensities.wavelength[i], intensities.flux[i])
# plt.plot(telluric.wavelength[i], telluric.flux[i])
# plt.show()

# # We use sme to estimate the limb darkening at each point
# # But use the extracted stellar spectrum, to create the actual intensities
intensities_combined = (intensities / stellar) * combined


# 9: Solve the equation system
i = np.arange(101)[sort][51]
wavelength = normalized[i].wavelength
flux = normalized[i].flux
planet_spectrum = load_planet()
nseg = normalized.segments.shape[0] - 1

for segment in tqdm(range(nseg)):
    spec, null = solve_prepared(
        normalized,
        telluric_space,
        combined,
        intensities_combined,
        detector,
        star,
        planet,
        seg=segment,
        solver="linear",
    )

    print("Saving data...")
    spec.write(join(done_dir, f"planet_extracted_{segment}.fits"))
    null.write(join(done_dir, f"null_extracted_{segment}.fits"))

    print("Plotting results...")
    spec._data = gaussian_filter1d(spec._data, nseg)
    null._data = gaussian_filter1d(null._data, nseg)
    spec = spec.resample(wavelength[segment], method="linear")
    null = null.resample(wavelength[segment], method="linear")

    ps = planet_spectrum.resample(wavelength[segment], "linear")

    plt.plot(
        wavelength[segment], flux[segment], label="normalized observation",
    )
    plt.plot(ps.wavelength, ps.flux, label="planet model")
    plt.plot(spec.wavelength, spec.flux, label="extracted")
    plt.xlim(wavelength[segment][0].value, wavelength[segment][-1].value)
    plt.ylim(0, 2)
    plt.ylabel("Flux, normalised")
    plt.xlabel("Wavelength [Å]")
    plt.legend()
    # plt.show()
    plt.savefig(join(done_dir, f"planet_spectrum_{segment}.png"))
    plt.clf()

    plt.plot(
        wavelength[segment], flux[segment], label="normalized observation",
    )
    plt.plot(ps.wavelength, ps.flux, label="planet model")
    plt.plot(null.wavelength, null.flux, label="extracted")
    plt.xlim(wavelength[segment][0].value, wavelength[segment][-1].value)
    plt.ylim(0, 2)
    plt.ylabel("Flux, normalised")
    plt.xlabel("Wavelength [Å]")
    plt.legend()
    # plt.show()
    plt.savefig(join(done_dir, f"null_spectrum_{segment}.png"))
    plt.clf()

    # Shift null spectrum unto spec spectrum
    sm = np.isfinite(spec.flux)
    sx, sy = spec.wavelength[sm], spec.flux[sm]
    nm = np.isfinite(null.flux)
    nx, ny = null.wavelength[nm], null.flux[nm]

    func = lambda x: (sy - np.interp(sx, nx * (1 - x / 3e5), ny)).value
    res = least_squares(func, x0=[-65], method="lm")
    rv = -res.x
    null = null.resample(null.wavelength * (1 - rv / 3e5))

    # Normalize to each other?
    # m = np.isfinite(spec.flux) & np.isfinite(null.flux)
    # sy, ny = spec.flux[m], null.flux[m]
    # func = lambda x: sy - (ny - x[0]) * x[1]
    # res = least_squares(func, x0=[0, 1], method="lm")
    # null.flux -= res.x[0]
    # null.flux *= res.x[1]

    # Take the difference between the two to get the planet spectrum?
    sflux = spec.flux - null.flux
    magnification = -1 / np.nanmin(sflux)
    # magnification = 2
    sflux = 1 + magnification * (spec.flux - null.flux)

    plt.plot(
        wavelength[segment], flux[segment], label="normalized observation",
    )
    plt.plot(ps.wavelength, ps.flux, label="planet model")
    plt.plot(spec.wavelength, sflux, label="extracted")
    plt.xlim(wavelength[segment][0].value, wavelength[segment][-1].value)
    plt.ylim(0, 2)
    plt.ylabel("Flux, normalised")
    plt.xlabel("Wavelength [Å]")
    plt.legend()
    # plt.show()
    plt.savefig(join(done_dir, f"diff_spectrum_{segment}.png"))
    plt.clf()

    spec._data = sflux
    spec.write(join(done_dir, f"diff_extracted_{segment}.fits"))

pass
