import inspect
import logging
import os
import warnings
from copy import deepcopy
from glob import glob
from os.path import basename, dirname, exists, join

from flex.static import write as flexwrite, read as flexread

import astropy.constants as const
import exoorbit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py
from astropy.utils.iers import IERS_Auto
from astropy import units as u
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.time import Time
from astropy.utils.data import conf
from cats.data_modules.stellar_db import StellarDb
from cats.data_modules.telluric_fit import TelluricFit
from cats.data_modules.telluric_model import TelluricModel
from cats.extractor.extract_stellar_parameters import (
    extract_stellar_parameters,
    first_guess,
    fit_observation,
)
from cats.extractor.extract_transit_parameters import extract_transit_parameters
from cats.extractor.normalize_observation import normalize_observation
from cats.extractor.prepare import create_intensities, create_stellar, create_telluric
from cats.pysysrem.sysrem import sysrem
from cats.simulator.detector import Crires
from cats.spectrum import Spectrum1D, SpectrumArray, SpectrumList
from exoorbit.bodies import Planet, Star
from pysme.sme import SME_Structure
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from scipy.optimize import least_squares
from tqdm import tqdm

from radtrans import radtrans
from solve_prepared import solve_prepared

conf.remote_timeout = 100.0
IERS_Auto()


logger = logging.getLogger(__name__)


class Step:
    def __init__(self, raw_dir, medium_dir, done_dir):
        self.raw_dir = raw_dir
        self.medium_dir = medium_dir
        self.done_dir = done_dir

    def run(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, *args, **kwargs):
        raise NotImplementedError

    def load(self, *args, **kwargs):
        raise NotImplementedError


class CollectObservationsStep(Step):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = None
        self.savefilename = join(self.medium_dir, "spectra.npz")

    def run(self, observatory, star, planet):
        files_fname = join(self.raw_dir, "*.fits")
        files = glob(files_fname)
        additional_data_fname = join(self.raw_dir, "*.csv")
        try:
            additional_data = glob(additional_data_fname)[0]
            additional_data = pd.read_csv(additional_data)
        except:
            additional_data = None

        speclist = []
        for f in tqdm(files):
            i = int(basename(f)[9:-5])
            hdu = fits.open(f)
            wave = hdu[1].data << u.AA
            flux = hdu[2].data << u.one

            if additional_data is not None:
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

        self.data = SpectrumArray(speclist)
        self.save(self.savefilename)
        return {"spectra": self.data}

    def save(self, fname=None):
        if self.data is None:
            raise ValueError("This step needs to be run first before data can be saved")
        if fname is None:
            fname = self.savefilename
        self.data.write(fname)

    def load(self, fname=None):
        if fname is None:
            fname = self.savefilename
        self.data = SpectrumArray.read(fname)
        return self.data


class CatsRunner:
    names_of_steps = {"collect": CollectObservationsStep}
    step_order = {"collect": 10}

    def __init__(
        self, detector, star, planet, raw_dir=None, medium_dir=None, done_dir=None
    ):
        self.detector = detector
        self.observatory = self.detector.observatory

        if not isinstance(star, Star):
            sdb = StellarDb()
            self.star = sdb.get(star)
        else:
            self.star = star

        if not isinstance(planet, Planet):
            self.planet = self.star.planets[planet]
        else:
            self.planet = planet

        self.orbit = exoorbit.Orbit(self.star, self.planet)

        if raw_dir is None:
            self.raw_dir = join(dirname(__file__), "raw")
        else:
            self.raw_dir = raw_dir

        if medium_dir is None:
            self.medium_dir = join(dirname(__file__), "medium")
        else:
            self.medium_dir = medium_dir

        if done_dir is None:
            self.done_dir = join(dirname(__file__), "done")
        else:
            self.done_dir = done_dir

    def run(self, steps):
        # Make sure the directories exists
        for d in [self.medium_dir, self.done_dir]:
            os.makedirs(d, exist_ok=True)

        if steps == "all":
            steps = list(self.step_order.keys())
        steps = list(steps)
        # Order steps in the best order
        steps = sorted(steps, key=lambda s: self.step_order[s])

        # Reset data
        self.data = {
            "raw_dir": self.raw_dir,
            "medium_dir": self.medium_dir,
            "done_dir": self.done_dir,
            "star": self.star,
            "planet": self.planet,
            "detector": self.detector,
            "observatory": self.observatory,
        }

        # Run individual steps
        for step in steps:
            self.run_module(step)

        return self.data

    def run_module(self, step, load=False):
        # The Module this step is based on (An object of the Step class)
        module = self.names_of_steps[step](self.raw_dir, self.medium_dir, self.done_dir)

        # Load the dependencies necessary for loading/running this step
        # We determine this through introspection
        members = inspect.getmembers(module.__class__.run)
        members = [m for m in members if m[0] == "__code__"][0][1]
        # We skip the first element, as that is 'self'
        dependencies = inspect.getargs(members).args[1:]
        # Then we get all the data from other steps if necessary
        for dependency in dependencies:
            if dependency not in self.data.keys():
                self.data[dependency] = self.run_module(dependency, load=True)
        args = {d: self.data[d] for d in dependencies}

        # Try to load the data, if the step is not specifically given as necessary
        # If the intermediate data is not available, run it normally instead
        # But give a warning
        if load:
            try:
                logger.info("Loading data from step '%s'", step)
                data = module.load(**args)
            except FileNotFoundError:
                logger.warning(
                    "Intermediate File(s) for loading step %s not found. Running it instead.",
                    step,
                )
                data = self.run_module(step, load=False)
        else:
            logger.info("Running step '%s'", step)
            data = module.run(**args)

        self.data[step] = data
        return data


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

# Detector
setting = "K/2/4"
detectors = [1, 2, 3]
orders = [7, 6, 5, 4, 3, 2]
detector = Crires(setting, detectors, orders=orders)

# Linelist
linelist = join(dirname(__file__), "crires_k_2_4.lin")

# Star info
star = "HD209458"
planet = "b"

raw_dir = join(dirname(__file__), "HD209458_v4")
runner = CatsRunner(detector, star, planet, raw_dir=raw_dir)
medium_dir = runner.medium_dir
done_dir = runner.done_dir

# Override data with known information
runner.star.vsini = 1.2 * (u.km / u.s)
runner.star.monh = 0 * u.one

# data = runner.run(["collect"])

# flexwrite("bla.flx", **data)
# data2 = flexread("bla.flx")

# spectra = data["collect"]

# 1: Collect observations
print("Collect observations")
fname = join(medium_dir, "spectra.npz")
cos = CollectObservationsStep(raw_dir, medium_dir, done_dir)
try:
    spectra = cos.load(fname)
except FileNotFoundError:
    files = join(raw_dir, "*.fits")
    spectra = cos.run(files, additional_data=additional_data)
    cos.save(fname)

times = spectra.datetime

flexwrite("bla.flex", spectra=spectra)
data2 = flexread("bla.flex")

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
planet = runner.planet
planet.t0 = p.t0
planet.inc = 86.59 * u.deg
planet.ecc = 0 * u.one
planet.period = 3.52472 * u.day
planet.radius = p.radius

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

fname = join(medium_dir, "reference_petitRADTRANS.fits")
if not exists(fname) or False:
    # Use petitRadtrans Code to create the reference spectrum
    wmin = 1 * u.um
    wmax = 3 * u.um
    ref = radtrans([wmin, wmax], star, planet)
    print("Done with PetitRadtrans")
    ref.write(fname)
else:
    ref = Spectrum1D.read(fname)

# Rescaled ref to 0 to 1
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

# How does SysRem help us?
# It removes the trend from the airmass, but then we can't use the tellurics anymore
# Unless we figure out the airmass that is being used by all the observations?


fname = join(medium_dir, "correlation.npz")
if not exists(fname) or False:
    # entries past 90 are 'weird'
    flux = normalized.flux.to_value(1)
    flux = flux[:90]
    unc = spectra.uncertainty.array[:90]

    correlation = {}
    for n in tqdm(range(101), desc="Sysrem N"):
        corrected_flux = sysrem(flux, num_errors=n, errors=unc)

        # Mask strong tellurics
        std = np.nanstd(corrected_flux, axis=0)
        std[std == 0] = 1
        corrected_flux /= std

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
        for i in tqdm(range(10), leave=False, desc="Sysrem on Cross Correlation"):
            correlation[f"{n}.{i}"] = sysrem(np.copy(corr), i)

    # Save the results
    np.savez(fname, **correlation)
    print(f"Saved cross-correlation to: {fname}")
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
# v_planet = -(v_sys + (v_planet << (u.km / u.s)))
v_planet = -v_planet << (u.km / u.s)


# TODO:
# For this dataset we can fit the lightcurve directly
# since all observations are the same, but in reallity that will be tricky
# So we should make sure that:
#   orb = Orbit(self.star, self.planet)
#   area = orb.stellar_surface_covered_by_planet(times)
# Works similarly well

# TODO: some of the area we calculate here should be explained by limb darkening

limits = 15, 85
f = spectra.flux.to_value(1)
y = np.nanmean(f, axis=1)
x = np.arange(len(y))

x2 = np.concatenate([x[: limits[0]], x[limits[1] :]])
y2 = np.concatenate([y[: limits[0]], y[limits[1] :]])
yf = np.polyval(np.polyfit(x2, y2, 3), x)

area = 1 - y / yf
area[: limits[0]] = area[limits[1] :] = 0
area = gaussian_filter1d(area, 1)

# 9: Solve the equation system
wavelength = normalized[51].wavelength
flux = normalized[51].flux
nseg = normalized.nseg

ref.flux[:] = gaussian_filter1d(ref.flux, 100)

cross_corr = {}
for seg in tqdm([13]):
    hspec = ref.resample(wavelength[seg], inplace=False)
    hspec.flux[:] -= np.nanmin(hspec.flux)
    hspec.flux[:] /= np.nanmax(hspec.flux)

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
    ]
    visible = [-1, -1]

    # TODO: put everything into one big extraction
    # for seg in tqdm(range(nseg)):
    # 10000000
    for regularization_weight in tqdm(
        [5000000], desc="Regularization Weight", leave=False
    ):
        d = []
        for n_sysrem in tqdm([10], desc="N Sysrem", leave=False):

            spec, null = solve_prepared(
                spectra,
                telluric,
                stellar_combined,
                spectra,
                detector,
                star,
                planet,
                solver="linear",
                seg=seg,
                rv=v_planet,
                n_sysrem=n_sysrem,
                regularization_weight=regularization_weight,
                regularization_ratio=10,
                area=area,
            )

            # print("Saving data...")
            spec.write(
                join(
                    done_dir,
                    f"planet_extracted_{seg}_{regularization_weight}_{n_sysrem}.fits",
                )
            )

            if n_sysrem is None:
                sflux = spec.flux - np.nanpercentile(spec.flux, 5)
                sflux /= np.nanpercentile(sflux, 95)
                nflux = null.flux - np.nanpercentile(null.flux, 5)
                nflux /= np.nanpercentile(nflux, 95)
            else:
                sflux = spec.flux
                nflux = null.flux

            d += [
                {
                    "x": spec.wavelength.to_value("AA"),
                    "y": sflux.to_value(1),
                    "name": f"extracted, RegWeight: {regularization_weight}, nSysrem: {n_sysrem}",
                },
                # {
                #     "x": null.wavelength.to_value("AA"),
                #     "y": nflux.to_value(1),
                #     "name": f"inversed, RegWeight: {regularization_weight}, nSysrem: {n_sysrem}",
                # },
            ]
            visible += [-1]

        minimum = min([np.nanpercentile(ds["y"], 5) for ds in d[0:]])
        for i in range(0, len(d)):
            d[i]["y"] -= minimum

        maximum = max([np.nanpercentile(ds["y"], 95) for ds in d[0:]])
        for i in range(0, len(d)):
            d[i]["y"] /= maximum

        for i in range(0, len(d)):
            dspec = np.interp(hspec.wavelength.to_value("AA"), d[i]["x"], d[i]["y"])
            cross_corr[f"{seg}.{i}"] = np.correlate(
                dspec, hspec.flux.to_value(1), mode="same"
            )

        data += d

    wran = [wavelength[seg][0].to_value("AA"), wavelength[seg][-1].to_value("AA")]
    layout = {
        "title": f"Segment: {seg}",
        "xaxis": {"title": "Wavelength [Å]", "range": wran},
        "yaxis": {"title": "Flux, normalised"},
    }
    fname = join(done_dir, f"planet_spectrum_{seg}.html")
    fig = go.Figure(data, layout)
    py.plot(fig, filename=fname, auto_open=False)

np.savez(join(done_dir, "cross_correlation.npz"), **cross_corr)
