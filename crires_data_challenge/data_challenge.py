import inspect
import logging
import os
from glob import glob
from os.path import basename, dirname, exists, join

import exoorbit
from exoorbit.bodies import Planet, Star

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py
from astropy import units as u
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.time import Time
from astropy.utils.iers import IERS_Auto
from cats.data_modules.stellar_db import StellarDb
from cats.data_modules.telluric_fit import TelluricFit
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
from flex.flex import FlexFile
from pysme.sme import SME_Structure
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from scipy.optimize import least_squares
from tqdm import tqdm

from radtrans import radtrans
from solve_prepared import solve_prepared

IERS_Auto()
logger = logging.getLogger(__name__)


class Step:
    def __init__(self, raw_dir, medium_dir, done_dir):
        self.raw_dir = raw_dir
        self.medium_dir = medium_dir
        self.done_dir = done_dir

    def run(self, *args, **kwargs):
        raise NotImplementedError


class StepIO:
    filename = None

    @property
    def savefilename(self):
        return join(self.medium_dir, self.filename)

    def save(self, data, filename=None):
        raise NotImplementedError

    def load(self, filename=None):
        raise NotImplementedError


class SpectrumArrayIO(StepIO):
    def save(self, data, fname=None):
        if fname is None:
            fname = self.savefilename
        data.write(fname)

    def load(self, fname=None):
        if fname is None:
            fname = self.savefilename
        data = SpectrumArray.read(fname)
        return data


class Spectrum1DIO(StepIO):
    def save(self, data, fname=None):
        if fname is None:
            fname = self.savefilename
        data.write(fname)

    def load(self, fname=None):
        if fname is None:
            fname = self.savefilename
        data = Spectrum1D.read(fname)
        return data


class SmeStructureIO(StepIO):
    def save(self, data, filename=None):
        if filename is None:
            filename = self.savefilename
        data.save(filename)

    def load(self, filename=None):
        if filename is None:
            filename = self.savefilename
        data = SME_Structure.load(filename)
        return data


class CollectObservationsStep(Step, SpectrumArrayIO):
    filename = "spectra.flex"

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

        data = SpectrumArray(speclist)
        self.save(data, self.savefilename)
        return data


class TelluricAirmassStep(Step, SpectrumArrayIO):
    filename = "telluric_airmass.npz"

    def run(self, spectra, star, observatory, detector):
        times = spectra.datetime
        wrange = detector.regions
        data = create_telluric(wrange, spectra, star, observatory, times)
        self.save(data, self.savefilename)
        return data


class TelluricTapasStep(Step, SpectrumArrayIO):
    filename = "telluric_tapas.npz"

    def run(self, spectra, star, observatory, detector):
        times = spectra.datetime
        wrange = detector.regions
        data = create_telluric(
            wrange, spectra, star, observatory, times, source="tapas"
        )
        self.save(data, self.savefilename)
        return data


class TelluricSpaceStep(Step, SpectrumArrayIO):
    filename = "telluric_space.npz"

    def run(self, spectra, star, observatory, detector):
        times = spectra.datetime
        wrange = detector.regions
        data = create_telluric(
            wrange, spectra, star, observatory, times, source="space"
        )
        self.save(data, self.savefilename)
        return data


class StellarParametersFirstGuessStep(Step, SmeStructureIO):
    filename = "parameters_first_guess.sme"

    def run(self, spectra, star, detector, linelist):
        blaze = detector.blaze
        sme = first_guess(spectra, star, blaze, linelist, detector)
        sme.cscale_flag = "fix"
        sme.vrad_flag = "fix"
        self.save(sme)
        return sme


class StellarParametersFitStep(Step, StepIO):
    filename = "star.yaml"

    def run(self, stellar_parameters_first_guess, star):
        sme = stellar_parameters_first_guess["sme"]
        sme, star = fit_observation(sme, star)
        self.save(star)
        return star

    def save(self, data, filename=None):
        if filename is None:
            filename = self.savefilename
        star.save(filename)

    def load(self, filename=None):
        if filename is None:
            filename = self.savefilename
        star = Star.load(filename)
        return star


class StellarSpectrumStep(Step, SpectrumArrayIO):
    filename = "stellar.npz"

    def run(self, star, detector, linelist, spectra):
        times = spectra.datetime
        wrange = detector.regions
        stellar = create_stellar(
            wrange, spectra, times, method="sme", star=star, linelist=linelist
        )
        self.save(stellar)
        return stellar


class NormalizeObservationsStep(Step, SpectrumArrayIO):
    filename = "normalized.flex"

    def run(self, spectra, stellar_spectrum, telluric, detector):
        normalized, stellar_broadening, telluric_broadening = normalize(
            spectra, stellar_spectrum, telluric, detector
        )
        normalized.meta["stellar_broadening"] = stellar_broadening
        normalized.meta["telluric_broadening"] = telluric_broadening
        self.save(normalized)
        broadening = {"stellar": stellar_broadening, "telluric": telluric_broadening}
        return normalized, broadening

    def load(self, filename=None):
        normalized = super().load(filename)
        broadening = {
            "stellar": normalized.meta["stellar_broadening"],
            "telluric": normalized.meta["telluric_broadening"],
        }
        return normalized, broadening


class StellarSpectrumCombinedStep(Step, SpectrumArrayIO):
    filename_stellar = "stellar_spectrum_combined.npz"
    filename_telluric = "telluric_combined.npz"

    def run(self, detector, normalized_observation, telluric, stellar_spectrum):
        wrange = detector.regions
        times = normalized_observation.datetime

        stellar_combined, telluric_combined = create_stellar(
            wrange,
            normalized_observation,
            times,
            method="combine",
            telluric=telluric,
            detector=detector,
            stellar=stellar_spectrum,
        )
        self.save(stellar_combined, self.filename_stellar)
        self.save(tellurics_combined, self.filename_tellurics)
        return stellar_combined, telluric_combined

    def load(self, filename_stellar=None, filename_telluric=None):
        if filename_stellar is None:
            filename_stellar = self.filename_stellar
        if filename_telluric is None:
            filename_telluric = self.filename_telluric

        stellar = super().load(filename_stellar)
        telluric = super().load(filename_telluric)
        return stellar, telluric


class PlanetParametersStep(Step, StepIO):
    filename = "planet.yaml"

    def run(self, spectra, telluric, star, planet):
        planet = extract_transit_parameters(spectra, telluric, star, planet)
        self.save(planet)
        return planet

    def save(self, data, filename=None):
        if filename is None:
            filename = self.savefilename
        star.save(filename)

    def load(self, filename=None):
        if filename is None:
            filename = self.savefilename
        star = Planet.load(filename)
        return star


class SpecificIntensitiesSmeStep(Step, SpectrumArrayIO):
    filename = "intensities.npz"

    def run(self, detector, spectra, star, planet, observatory, linelist):
        wrange = detector.regions
        intensities = create_intensities(
            wrange, spectra, star, planet, observatory, times, linelist
        )
        self.save(intensities)
        return intensities


class PlanetAtmosphereReferencePetitRadtransStep(Step, Spectrum1DIO):
    filename = "reference_petitRADTRANS.npz"

    def run(self, star, planet, detector):
        wrange = detector.regions
        wmin = min([wr.lower for wr in wrange])
        wmax = max([wr.upper for wr in wrange])
        # Apply possible radial velocity tolerance of 200 km/s
        wmin *= 1 - 200 / 3e5
        wmax *= 1 + 200 / 3e5
        ref = radtrans([wmin, wmax], star, planet)
        ref.star = star
        ref.planet = planet

        # Rescaled ref to 0 to 1
        f = np.sqrt(1 - ref.flux)
        f -= f.min()
        f /= f.max()
        f = 1 - f ** 2
        ref.flux[:] = f

        self.save(ref)
        return ref


class CrossCorrelationReferenceStep(Step, Spectrum1DIO):
    filename = "reference_petitRADTRANS.fits"

    def run(self, planet_reference_spectrum, star, observatory, spectra):
        rv_range = 100
        rv_points = 201

        ref = planet_reference_spectrum
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
        # We are only looking at the difference between the median and the observation
        # Thus additional absorption would result in a negative signal at points of large absorption
        reference.flux[:] -= 1

        self.save(reference)
        return reference


class CrossCorrelationStep(Step, StepIO):
    filename = "cross_correlation.npz"

    def run(self, spectra, cross_correlation_reference):
        max_nsysrem = 10
        rv_range = 100
        rv_points = 201

        reference = cross_correlation_reference

        # entries past 90 are 'weird'
        flux = spectra.flux.to_value(1)
        flux = flux[:90]
        unc = spectra.uncertainty.array[:90]

        correlation = {}
        for n in tqdm(range(max_nsysrem), desc="Sysrem N"):
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

        self.save(correlation)
        return correlation

    def save(self, data, filename=None):
        if filename is None:
            filename = self.savefilename
        np.savez(filename, **data)

    def load(self, filename=None):
        if filename is None:
            filename = self.savefilename
        data = np.load(filename)
        return data


class PlanetRadialVelocityStep(Step, StepIO):
    filename = "planet_radial_velocity.flex"

    def run(self, cross_correlation, spectra):
        rv_range = 100
        rv_points = 201
        n, i = 2, 4

        corr = cross_correlation[f"{n}.{i}"]
        # Remove large scale variation
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

        data = {"rv_system": v_sys, "rv_planet": v_planet}

        self.save(data)
        return data

    def save(self, data, filename=None):
        if filename is None:
            filename = self.savefilename
        ff = FlexFile(header=data)
        ff.write(filename)

    def load(self, filename=None):
        if filename is None:
            filename = self.savefilename
        ff = FlexFile.read(filename)
        return ff.header


class PlanetAbsorptionAreaStep(Step, StepIO):
    filename = "planet_area.npy"

    def run(self, spectra):
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
        self.save(area)
        return area

    def save(self, data, filename=None):
        if filename is None:
            filename = self.savefilename
        np.save(filename, data)

    def load(self, filename=None):
        if filename is None:
            filename = self.savefilename
        data = np.load(filename)
        return data


class SolveProblemStep(Step):
    def run(
        self,
        normalized_observation,
        planet_reference_spectrum,
        spectra,
        detector,
        star,
        planet,
        planet_radial_velocity,
        planet_area,
    ):
        normalized, broadening = normalized_observation
        v_planet = planet_radial_velocity["rv_planet"]
        wavelength = normalized[51].wavelength
        flux = normalized[51].flux
        nseg = normalized.nseg

        planet_reference_spectrum.flux[:] = gaussian_filter1d(
            planet_reference_spectrum.flux, 100
        )

        return_data = {}

        for seg in tqdm(range(nseg)):
            hspec = planet_reference_spectrum.resample(wavelength[seg], inplace=False)
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

            for regularization_weight in tqdm(
                [5000000], desc="Regularization Weight", leave=False
            ):
                d = []
                for n_sysrem in tqdm([10], desc="N Sysrem", leave=False):

                    spec, null = solve_prepared(
                        spectra,  # (normalized) observation
                        spectra,  # Stellar spectrum
                        spectra,  # Tellurics
                        spectra,  # specific intensities
                        detector,
                        star,
                        planet,
                        solver="linear",
                        seg=seg,
                        rv=v_planet,
                        n_sysrem=n_sysrem,
                        regularization_weight=regularization_weight,
                        regularization_ratio=10,
                        area=planet_area,
                    )

                    return_data[seg] = spec

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
                    ]
                    visible += [-1]

                # Normalize
                minimum = min([np.nanpercentile(ds["y"], 5) for ds in d[0:]])
                for i in range(0, len(d)):
                    d[i]["y"] -= minimum

                maximum = max([np.nanpercentile(ds["y"], 95) for ds in d[0:]])
                for i in range(0, len(d)):
                    d[i]["y"] /= maximum

                for i in range(0, len(d)):
                    dspec = np.interp(
                        hspec.wavelength.to_value("AA"), d[i]["x"], d[i]["y"]
                    )

                data += d

            wran = [
                wavelength[seg][0].to_value("AA"),
                wavelength[seg][-1].to_value("AA"),
            ]
            layout = {
                "title": f"Segment: {seg}",
                "xaxis": {"title": "Wavelength [Å]", "range": wran},
                "yaxis": {"title": "Flux, normalised"},
            }
            fname = join(self.done_dir, f"planet_spectrum_{seg}.html")
            fig = go.Figure(data, layout)
            py.plot(fig, filename=fname, auto_open=False)

        return return_data


class CatsRunner:
    names_of_steps = {
        "spectra": CollectObservationsStep,
        "telluric": TelluricAirmassStep,
        "telluric_tapas": TelluricTapasStep,
        "telluric_space": TelluricSpaceStep,
        "stellar_parameters_first_guess": StellarParametersFirstGuessStep,
        "star": StellarParametersFitStep,
        "stellar_spectrum": StellarSpectrumStep,
        "normalized_observation": NormalizeObservationsStep,
        "stellar_spectrum_combined": StellarSpectrumCombinedStep,
        "planet": PlanetParametersStep,
        "specific_intensities": SpecificIntensitiesSmeStep,
        "planet_reference_spectrum": PlanetAtmosphereReferencePetitRadtransStep,
        "cross_correlation_reference": CrossCorrelationReferenceStep,
        "cross_correlation": CrossCorrelationStep,
        "planet_radial_velocity": PlanetRadialVelocityStep,
        "planet_area": PlanetAbsorptionAreaStep,
        "solve_problem": SolveProblemStep,
    }
    step_order = {
        "spectra": 10,
        "telluric_airmass": 20,
        "telluric_tapas": 30,
        "telluric_space": 40,
        "stellar_parameters_first_guess": 50,
        "star": 60,
        "stellar_spectrum": 70,
        "normalized_observation": 80,
        "stellar_spectrum_combined": 90,
        "planet": 100,
        "specific_intensities": 110,
        "planet_reference_spectrum": 120,
        "cross_correlation_reference": 130,
        "cross_correlation": 140,
        "planet_radial_velocity": 150,
        "planet_area": 160,
        "solve_problem": 500,
    }

    def __init__(
        self,
        detector,
        star,
        planet,
        linelist,
        raw_dir=None,
        medium_dir=None,
        done_dir=None,
    ):
        #:Detector: The detector/instrument performing the measurements
        self.detector = detector
        #:EarthLocation: The observatory coordinates
        self.observatory = self.detector.observatory

        if not isinstance(star, Star):
            sdb = StellarDb()
            star = sdb.get(star)
        #:Star: The parameters of the star
        self.star = star

        if not isinstance(planet, Planet):
            planet = self.star.planets[planet]
        #:Planet: The parameters of the planet
        self.planet = planet

        self.linelist = linelist
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
            "linelist": self.linelist,
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
                data = module.load()
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

    return normalized, stellar_broadening, telluric_broadening


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
runner = CatsRunner(detector, star, planet, linelist, raw_dir=raw_dir)

# Override data with known information
runner.star.vsini = 1.2 * (u.km / u.s)
runner.star.monh = 0 * u.one
runner.star.name = "HD209458"
runner.star.radial_velocity = -14.743 * (u.km / u.s)

runner.planet.inc = 86.59 * u.deg
runner.planet.ecc = 0 * u.one
runner.planet.period = 3.52472 * u.day

# Run the Runner
data = runner.run(["solve_problem"])
pass

# TODO:
# Adjust the mask manually
