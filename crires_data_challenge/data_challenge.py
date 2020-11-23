from glob import glob
import os
from os.path import basename, dirname, exists, join
from copy import deepcopy

import plotly.offline as py
import plotly.graph_objs as go
from plotly.io import write_image

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
from scipy.interpolate import interp1d
import astropy.constants as const
from scipy.signal import find_peaks

from astropy.nddata import StdDevUncertainty

import exoorbit
from cats.data_modules.telluric_fit import TelluricFit
from scipy.ndimage.morphology import binary_erosion, binary_dilation

from cats.data_modules.stellar_db import StellarDb
from cats.data_modules.telluric_model import TelluricModel
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
from hitran_linelist import Hitran, HitranSpectrum
from planet_spectra import load_planet
from radtrans import radtrans

from tellurics.tellurics import TapasTellurics


import warnings
from astroplan.utils import (
    download_IERS_A,
    _get_IERS_A_table,
)
from astropy.utils.data import conf

conf.remote_timeout = 100.0

# Make sure we have an up to date IERS_A table
needs_download = False
try:
    with warnings.catch_warnings(record=True) as warn:
        _get_IERS_A_table()
        if len(warn) != 0:
            # Table is out of date
            needs_download = True
except OSError:
    # Table is not cached
    needs_download = True

if needs_download:
    download_IERS_A()


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

                # We just assume shot noise, no read out noise etc
                unc = np.sqrt(np.abs(f))
                unc = StdDevUncertainty(unc)
                spec = Spectrum1D(
                    flux=f,
                    spectral_axis=w,
                    uncertainty=unc,
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
    stellar_broadening = 1
    telluric_broadening = 1

    for _ in tqdm(range(3), leave=False, desc="Iteration"):
        normalized = normalize_observation(
            spectra,
            stellar,
            telluric,
            detector,
            stellar_broadening,
            telluric_broadening,
        )

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

    mask = coeff[:, 0] > 0
    # mask = ~binary_erosion(~mask)
    # mask = ~binary_dilation(~mask, iterations=5)
    # Use morphology to improve mask
    for i in np.arange(coeff.shape[0])[mask]:
        coeff[i] = np.polyfit(airmass, telluric.flux[:, i], 1)

    coeff = t.spline_fit(coeff, s=1)
    model = t.model(coeff, airmass)
    model = np.clip(model, 0, 1)
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
star.monh = 0 * u.one
planet = star.planets["b"]

orbit = exoorbit.Orbit(star, planet)
t = [orbit.first_contact(), orbit.fourth_contact()]
rv = orbit.radial_velocity_planet(t)

# Data locations
raw_dir = join(dirname(__file__), "HD209458_v4")
medium_dir = join(dirname(__file__), "medium")
done_dir = join(dirname(__file__), "done")

for d in [medium_dir, done_dir]:
    os.makedirs(d, exist_ok=True)


# Other data
linelist = join(dirname(__file__), "crires_k_2_4.lin")
additional_data = join(raw_dir, "HD209458_additional_data.csv")

# 1: Collect observations
print("Collect observations")
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
print("Create telluric guess")
fname = join(medium_dir, "telluric.npz")
if not exists(fname) or False:
    telluric = create_telluric(wrange, spectra, star, observatory, times)
    telluric.write(fname)
else:
    telluric = SpectrumArray.read(fname)

fname = join(medium_dir, "telluric_tapas.npz")
if not exists(fname) or False:
    telluric_tapas = create_telluric(
        wrange, spectra, star, observatory, times, source="tapas"
    )
    # telluric_tapas = TapasTellurics(star, observatory)
    # telluric_tapas = telluric_tapas.get(detector.regions, spectra.datetime)
    # telluric_tapas = telluric_tapas.resample(spectra.wavelength, inplace=False)
    telluric_tapas.write(fname)
else:
    telluric_tapas = SpectrumArray.read(fname)

fname = join(medium_dir, "telluric_space.npz")
if not exists(fname) or False:
    telluric_space = create_telluric(
        wrange, spectra, star, observatory, times, source="space"
    )
    telluric_space.write(fname)
else:
    telluric_space = SpectrumArray.read(fname)

# 4: Extract stellar parameters
print("Determine stellar parameters")
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
star.name = "HD209458"
star.radial_velocity = -14.743 * (u.km / u.s)


# 5: Create stellar spectra
print("Calculate stellar spectra")
fname = join(medium_dir, "stellar.npz")
if not exists(fname) or False:
    stellar = create_stellar(
        wrange, spectra, times, method="sme", star=star, linelist=linelist
    )
    stellar.write(fname)
else:
    stellar = SpectrumArray.read(fname)


# 6: Normalize observations
# TODO: This needs work, as it does not do what we want it to do
# It appears to remove the planet signal, unfortunately
print("Normalize observations")
fname = join(medium_dir, "normalized.npz")
fname2 = join(medium_dir, "broadening_values.npz")
if not exists(fname) or not exists(fname2) or False:
    normalized, detector, stellar_broadening, telluric_broadening = normalize(
        spectra, stellar, telluric, detector
    )
    normalized.write(fname)
    np.savez(
        fname2,
        stellar_broadening=stellar_broadening,
        telluric_broadening=telluric_broadening,
    )
else:
    normalized = SpectrumArray.read(fname)
    data = np.load(fname2)
    stellar_broadening = data["stellar_broadening"]
    telluric_broadening = data["telluric_broadening"]


detector.spectral_broadening = stellar_broadening
telluric.flux = gaussian_filter1d(telluric.flux, telluric_broadening) << u.one
stellar.flux = gaussian_filter1d(stellar.flux, stellar_broadening) << u.one

# Create combined observations
fname = join(medium_dir, "stellar_combined.npz")
fname2 = join(medium_dir, "telluric_combined.npz")

if not exists(fname) or not exists(fname2) or False:
    stellar_combined, telluric_combined = create_stellar(
        wrange,
        normalized,
        times,
        method="combine",
        mask=sme.mask_cont,
        telluric=telluric,
        detector=detector,
        stellar=stellar,
    )
    stellar_combined.write(fname)
    telluric_combined.write(fname2)
else:
    stellar_combined = SpectrumArray.read(fname)
    telluric_combined = SpectrumArray.read(fname2)

# 7: Determine Planet transit
print("Determine the planet transit parameters")
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
# telluric = fit_tellurics(
#     normalized, telluric, star, observatory, skip_resample=True, degree=1
# )

# 8: Create specific intensitiies
fname = join(medium_dir, "intensities.npz")
if not exists(fname) or False:
    intensities = create_intensities(
        wrange, spectra, star, planet, observatory, times, linelist
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
# plt.plot(combined.wavelength[i], combined.flux[i])
# plt.plot(stellar.wavelength[i], stellar.flux[i])
# plt.plot(intensities.wavelength[i], intensities.flux[i])
# plt.plot(telluric.wavelength[i], telluric.flux[i])
# plt.show()

# # We use sme to estimate the limb darkening at each point
# # But use the extracted stellar spectrum, to create the actual intensities
intensities_combined = (intensities / stellar) * stellar_combined


# TODO: is this wavelength scale in air or in vacuum????
# It's the same as telluric, so it should be in air!
# but CRIRES is in vacuum!
fname = join(medium_dir, "reference_petitRADTRANS.fits")
if not exists(fname) or False:
    # Use planet model as reference
    # TODO: use petitRadtrans Code to create the reference spectrum
    # Currently we use the input spectrum!!!
    # ref = HitranSpectrum()
    wmin = 1 * u.um
    wmax = 3 * u.um
    ref = radtrans([wmin, wmax], star, planet)
    print("Done with PetitRadtrans")
    ref.write(fname)
else:
    ref = Spectrum1D.read(fname)

# from cats.utils import air2vac, vac2air

from specutils.utils.wcs_utils import vac_to_air, air_to_vac

# Rescaled to 0 to 1
f = np.sqrt(1 - ref.flux)
f -= f.min()
f /= f.max()
f = 1 - f ** 2

ref = Spectrum1D(spectral_axis=ref.wavelength, flux=f, reference_frame="barycentric")

fname = join(medium_dir, "reference_telluric.fits")
if not exists(fname) or False:
    # # Use airmass 2 spectrum as reference
    tmodel = TelluricModel(star, observatory)
    ref_tell = tmodel.interpolate_spectra(2)
    ref_tell.write(fname)
else:
    ref_tell = Spectrum1D.read(fname)

# ref_tell = Spectrum1D(flux=telluric.flux[50], spectral_axis=telluric.wavelength[50])
# ref_tell.flux[:] = ref_tell.flux * stellar.flux[50]
# # Scale telluric spectrum to the same scale as the model flux
# rmin = ref.flux.min()
# rmax = ref.flux.max()
# ref_tell.flux[:] = ref_tell.flux * (rmax - rmin) + rmin
# ref = ref_tell

rv_range = 100
rv_points = 201

fname = join(medium_dir, "corr_reference.npz")
if not exists(fname) or False:
    ref.star = star
    ref.observatory_location = observatory
    ref.datetime = spectra.datetime[50]
    ref_wave = np.copy(spectra.wavelength[50])
    reference = np.zeros((rv_points, ref_wave.size))

    rv = np.linspace(-rv_range, rv_range, num=rv_points)
    rv = rv << (u.km / u.s)

    for i in tqdm(range(rv_points)):
        tmp = ref.shift("barycentric", rv=rv[i], inplace=False)
        tmp = tmp.resample(ref_wave, inplace=False, method="linear")
        reference[i] = np.nan_to_num(tmp.flux.to_value(1))

    reference = reference << u.one
    reference = Spectrum1D(
        spectral_axis=ref_wave, flux=reference, datetime=spectra.datetime[50]
    )
    reference.write(fname)
else:
    reference = Spectrum1D.read(fname)

# We are only looking at the difference between the median and the observation
# Thus additional absorption would result in a negative signal at points of large absorption
reference.flux[:] -= 1

# Add absorption for fun
# for i in range(20, 80):
#     normalized.flux[i] -= 1e-3 * reference.flux[-12 + i]

# TODO: How would SysRem help us?
# It removes the trend from the airmass, but then we can't use the tellurics anymore
# Unless we figure out the airmass that is being used by all the observations?
from cats.pysysrem.sysrem import sysrem


fname = join(medium_dir, "correlation.npz")
if not exists(fname) or False:
    # # Mask strong telluric absorbtion (and nan values)
    # mask = telluric.flux.to_value(1) >= 0.1
    # # but not nans at the side?
    # # TODO: have tellurics without nans at the border
    # mask |= np.isnan(telluric.flux.to_value(1))
    # mask = mask[:90]
    # mask |= np.any(mask, axis=0)

    # flux = normalized.shift("barycentric", inplace=False)
    # flux = flux.resample(spectra.wavelength[50], inplace=True)
    flux = normalized.flux.to_value(1)
    flux = flux[:90]
    unc = spectra.uncertainty.array[:90]

    correlation = {}
    for n in tqdm(range(3), desc="Sysrem N"):
        corrected_flux = sysrem(flux, num_errors=n, iterations=10, errors=unc)
        # Mask strong tellurics
        # corrected_flux[~mask] = np.nan
        # corrected_flux -= gaussian_filter1d(corrected_flux, 20, axis=1)
        std = np.nanstd(corrected_flux, axis=0)
        std[std == 0] = 1
        corrected_flux /= std
        # corrected_flux += np.median(normalized.flux.to_value(1), axis=0)
        # normalized_sysrem.flux = corrected_flux << normalized.flux.unit

        # Observations 90 to 101 have weird stuff
        corr = np.zeros((90, rv_points))
        for i in tqdm(range(90), leave=False, desc="Observation"):
            for j in tqdm(range(rv_points), leave=False, desc="radial velocity",):
                for left, right in zip(spectra.segments[:-1], spectra.segments[1:]):
                    m = np.isnan(corrected_flux[i, left:right])
                    m |= np.isnan(reference.flux[j, left:right].to_value(1))
                    m = ~m
                    # Cross correlate!
                    corr[i, j] += np.correlate(
                        corrected_flux[i, left:right][m],
                        reference.flux[j, left:right][m].to_value(1),
                        "valid",
                    )
                    # Normalize to the number of data points used
                    corr[i, j] *= m.size / np.count_nonzero(m)

        correlation[f"{n}"] = np.copy(corr)
        for i in tqdm(range(5), leave=False, desc="Sysrem on Cross Correlation"):
            correlation[f"{n}.{i}"] = sysrem(np.copy(corr), i)

    # Save the results
    np.savez(fname, **correlation)
else:
    correlation = np.load(fname)


n, i = 2, 4
corr = correlation[f"{n}.{i}"]
# vmin, vmax = np.nanpercentile(correlation[f"{n}.{i}"], (5, 95))
# plt.imshow(correlation[f"{n}.{i}"], aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
# plt.title(f"N Sysrem: {n}")
# plt.xticks(
#     np.linspace(0, rv_points, 21), labels=np.linspace(-rv_range, rv_range, 21),
# )
# plt.xlabel("v [km/s]")
# plt.ylabel("n observation")
# plt.show()

tmp = np.copy(corr)
tmp -= gaussian_filter1d(corr, 20, axis=0)
tmp -= gaussian_filter1d(corr, 20, axis=1)
tmp -= np.nanmedian(tmp)
corr = tmp

# Fit the detected cross correlation signal with a model
# TODO: find decent initial values on your own
# TODO: maybe use MCMC?
n_obs = spectra.shape[0]
A = np.nanpercentile(corr, 99)
# This starting value is very important!!!
# v_sys = star.radial_velocity.to_value("km/s")
v_sys = -35  # Star radial velocity + barycentric correction
v_planet = 30 / 60
sig = 2
lower, upper = 20, 80
x0 = [v_sys, v_planet, sig, A]
x = np.linspace(-rv_range, rv_range + 1, rv_points)


def gaussian(x, A, mu, sig):
    return A * np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))


def model_func(x0):
    mu, shear, sig, A = x0
    model = np.zeros_like(corr)
    for i in range(lower, upper):
        mu_prime = mu + shear * (i - n_obs // 2)
        model[i] = gaussian(x, A, mu_prime, sig)
    return model


def fitfunc(x0):
    model = model_func(x0)
    resid = model - corr
    return resid.ravel()


res = least_squares(
    fitfunc,
    x0=x0,
    loss="soft_l1",
    bounds=[[-rv_range, 0, 1, 1], [rv_range, 2, 5, 200]],
    x_scale="jac",
    ftol=None,
)
model = model_func(res.x)
v_sys = res.x[0] << (u.km / u.s)
v_planet = res.x[1] * (np.arange(n_obs) - n_obs // 2)
v_planet = v_sys + (v_planet << (u.km / u.s))

# ax1 = plt.subplot(2, 1, 1)
# plt.imshow(corr, aspect="auto", origin="lower")
# plt.subplot(212, sharex=ax1)
# plt.imshow(model, aspect="auto", origin="lower")
# plt.show()

# tmp = np.copy(corr)
# tmp -= gaussian_filter1d(corr, 20, axis=0)
# tmp -= gaussian_filter1d(corr, 20, axis=1)

# ax1 = plt.subplot(2, 1, 1)
# vmin, vmax = np.nanpercentile(corr, (5, 95))
# plt.imshow(corr, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
# plt.subplot(212, sharex=ax1)
# vmin, vmax = np.nanpercentile(tmp, (5, 95))
# plt.imshow(tmp, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
# plt.show()

# vmin, vmax = np.nanpercentile(corrected_flux, (5, 95))
# plt.imshow(corrected_flux, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
# plt.show()

# Shifting the observations and stacking them doesn't work (as expected?)
# normalized_sysrem = deepcopy(normalized)
# normalized_sysrem.flux[:] = sysrem(normalized_sysrem, 1)
# normalized_sysrem.flux[:] /= np.nanstd(normalized_sysrem.flux, axis=0)


# TODO: calculate the planet size from this plot
# plt.plot(np.median(corrected_flux, axis=1))
# plt.show()


# telluric_corrected_flux = sysrem(telluric_combined.flux.to_value(1))
# telluric_combined.flux = telluric_corrected_flux << telluric_combined.flux.unit

# stellar_corrected_flux = sysrem(stellar_combined.flux.to_value(1))
# stellar_combined.flux = stellar_corrected_flux << stellar_combined.flux.unit

# 9: Solve the equation system
i = np.arange(101)[sort][51]
wavelength = normalized[i].wavelength
flux = normalized[i].flux
planet_spectrum = load_planet()
nseg = normalized.nseg

# hitspec = HitranSpectrum()
# hitspec.datetime = normalized.datetime[i]
# hitspec = hitspec.shift(normalized.reference_frame)

# import spectres

# hflux = spectres.spectres(
#     normalized.wavelength[i].to_value("AA")[::10],
#     hitspec.wavelength.to_value("AA"),
#     hitspec.flux.to_value(1),
# )
# hflux -= np.nanmin(hflux)
# hflux /= np.nanmax(hflux)

# hitspec = Spectrum1D(spectral_axis=normalized.wavelength[i][::10], flux=hflux << u.one)
# hitspec = ref.shift("telescope", rv=-v_sys)
# hitspec = hitspec.resample(spectra.wavelength[50])

hitspec = ref.shift("barycentric", rv=v_sys, inplace=False)
# hitspec = hitspec.resample(normalized.wavelength[i], method="flux_conserving")


spec_all, null_all = solve_prepared(
    normalized,
    telluric_combined,
    stellar_combined,
    intensities,
    detector,
    star,
    planet,
    solver="linear",
    rv=v_planet,
)

# print("Saving data...")
# spec.write(join(done_dir, f"planet_extracted.fits"))
# null.write(join(done_dir, f"null_extracted.fits"))

# TODO: put everything into one big extraction
for seg in tqdm(range(nseg)):
    print("Plotting results...")
    # spec._data = gaussian_filter1d(spec._data, nseg)
    # null._data = gaussian_filter1d(null._data, nseg)
    # spec = spec.resample(wavelength[segment], method="linear")
    # null = null.resample(wavelength[segment], method="linear")

    # spec, null = solve_prepared(
    #     normalized,
    #     telluric_combined,
    #     stellar_combined,
    #     intensities,
    #     detector,
    #     star,
    #     planet,
    #     solver="linear",
    #     seg=seg,
    #     rv=v_planet,
    # )
    lower, upper = spectra.segments[seg:seg+2]
    spec, null = spec_all[lower:upper], null_all[lower:upper]

    spec.star = null.star = star
    spec.planet = null.planet = planet
    spec.observatory_location = null.observatory_location = observatory
    spe.datetime = null.datetime = spectra.datetime[50]

    print("Saving data...")
    spec.write(join(done_dir, f"planet_extracted_{seg}.fits"))
    null.write(join(done_dir, f"null_extracted_{seg}.fits"))

    spec.flux[:40] = spec.flux[-60:] = np.nan
    null.flux[:40] = null.flux[-60:] = np.nan

    # Shift null spectrum unto spec spectrum
    sm = np.isfinite(spec.flux)
    sx, sy = (
        spec.wavelength[sm].to_value("AA"),
        gaussian_filter1d(spec.flux[sm].to_value(1), nseg),
    )
    nm = np.isfinite(null.flux)
    nx, ny = (
        null.wavelength[nm].to_value("AA"),
        gaussian_filter1d(null.flux[nm].to_value(1), nseg),
    )
    c_light = const.c.to_value("km/s")

    def func(x):
        beta = x[0] / c_light
        shifted = nx * np.sqrt((1 + beta) / (1 - beta))
        ni = interp1d(
            shifted, (ny - x[1]) * x[2], kind="slinear", fill_value="extrapolate"
        )(sx)
        return gaussian_filter1d(sy - ni, nseg)

    ms, mn = np.median(sy), np.median(ny)
    res = least_squares(func, x0=[65.0, ms - mn, ms / mn], method="trf", loss="soft_l1")
    rv = res.x[0] << (u.km / u.s)

    null._data -= res.x[1]
    null._data *= res.x[2]
    null = null.shift("planet", rv=rv)
    null = null.resample(spec.wavelength)

    sflux = spec.flux - np.nanpercentile(spec.flux, 5)
    sflux /= np.nanpercentile(sflux, 95)
    nflux = null.flux - np.nanpercentile(null.flux, 5)
    nflux /= np.nanpercentile(null.flux, 95)

    hspec = hitspec.resample(wavelength[seg], inplace=False)
    hspec.flux[:] -= np.nanmin(hspec.flux)
    hspec.flux[:] /= np.nanmax(hspec.flux)

    # Plot plotly
    data = [
        {
            "x": wavelength[seg].to_value("AA"),
            "y": flux[seg].to_value(1),
            "name": "normalized observation",
        },
        {
            "x": wavelength[seg].to_value("AA"),
            "y": hspec.flux.to_value(1),
            "name": "planet model",
        },
        {
            "x": spec.wavelength.to_value("AA"),
            "y": sflux.to_value(1),
            "name": "extracted",
        },
        {
            "x": null.wavelength.to_value("AA"),
            "y": nflux.to_value(1),
            "name": "extracted (reverse rv)",
        },
    ]

    # Take the difference between the two to get the planet spectrum?
    # null.wavelenth != spec.wavelnegth BUT the spectra are aligned?
    sflux = spec.flux - null.flux
    magnification = -1 / np.nanpercentile(sflux, 1)
    sflux = 1 + magnification * sflux
    wave = spec.wavelength

    data += [
        {
            "x": wave.to_value("AA"),
            "y": sflux.to_value(1),
            "name": "extracted_difference",
        }
    ]

    wran = [wavelength[seg][0].to_value("AA"), wavelength[seg][-1].to_value("AA")]
    layout = {
        "title": f"Segment: {seg}; Regularization weight: {spec.meta['regularization_weight']}",
        "xaxis": {"title": "Wavelength [Å]", "range": wran},
        "yaxis": {"title": "Flux, normalised", "range": [0, 2]},
    }
    fname = join(done_dir, f"planet_spectrum_{seg}.html")
    fig = go.Figure(data, layout)
    py.plot(fig, filename=fname, auto_open=False)
pass
