"""
Option A:
  - resample all spectra to a common wavelength grid
  - add them together
  - fit to that
Pros:
  - easier to normalize
  - radial velocity shift between observations is small (if from the same transit)
Cons:
  - Loose some precision due to resampling

Option B:
  - calculate SME spectra without sampling to wavelength grid
  - do least-squares to each spectrum individually and add them together
Pros:
  - more precise, since no interpolation
Cons:
  - difficult to normalize individual spectra
  - more effort

Lets do Option A!

"""
from glob import glob
from os.path import dirname, join

import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

import astroplan as ap
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.constants import c
from pysme.linelist.vald import ValdFile
from pysme.sme import SME_Structure
from pysme.solve import SME_Solver
from pysme.synthesize import synthesize_spectrum, Synthesizer
from pysme.gui import plot_plotly
from tqdm import tqdm

from scipy.optimize import curve_fit
from exoorbit.bodies import Star

from ..data_modules.stellar_db import StellarDb
from ..simulator.detector import Crires
from ..spectrum import SpectrumArray
from ..data_modules.combine import combine_observations


def round_to_nearest(value: np.ndarray, options: list):
    value = np.atleast_2d(value).T
    options = np.asarray(options)

    diff = np.abs(value - options)
    sort = np.argsort(diff)
    nearest = options[sort[:, 0]]
    if value.size == 1:
        return nearest[0]
    return nearest


def detect_ouliers(spectra: SpectrumArray):
    flux = np.copy(spectra.flux)
    for i in range(len(spectra)):
        flux[i] /= np.nanpercentile(flux[i], 95)

    median = np.nanmedian(flux, axis=0)
    flux = np.abs(flux - median)
    mad = np.nanmedian(flux, axis=0)
    mask = flux > 5 * mad
    mask |= np.isnan(spectra.flux)

    flux = np.ma.array(spectra.flux, mask=mask)
    spectrum = np.ma.mean(flux, axis=0)
    uncs = np.ma.std(flux, axis=0)
    return spectrum, uncs


def create_first_guess(
    spectrum: SpectrumArray,
    star: Star,
    blaze: np.ndarray,
    linelist: str,
    detector,
    uncs = None,
):
    print("Extracting stellar parameters...")

    # Create SME structure
    print("Preparing PySME structure")
    sme = SME_Structure()
    sme.wave = [wave.to_value(u.AA) for wave in spectrum.wavelength]
    sme.spec = [spec.to_value(1) for spec in spectrum.flux]
    if spectrum.uncertainty is not None:
        sme.uncs = [unc.array * unc.unit.to(1) for unc in spectrum.uncertainty]

    sme.teff = star.teff.to_value(u.K)
    sme.logg = star.logg.to_value(1)
    sme.monh = star.monh.to_value(1)
    sme.vturb = star.vturb.to_value(u.km / u.s)

    sme.abund = "solar"
    sme.linelist = ValdFile(linelist)

    sme.atmo.source = "marcs"
    sme.atmo.method = "grid"

    nlte = None
    if nlte is not None:
        for elem, grid in nlte.items():
            sme.nlte.set_nlte(elem, grid)

    sme.cscale_flag = "none"
    sme.normalize_by_continuum = True
    sme.vrad_flag = "fix"
    sme.vrad = star.radial_velocity.to_value("km/s")

    if detector is not None:
        sme.iptype = "gauss"
        sme.ipres = detector.resolution

    # Create an initial spectrum using the nominal values
    # This also determines the radial velocity
    print("Determine the radial velocity using the nominal stellar parameters")
    synthesizer = Synthesizer()
    sme = synthesizer.synthesize_spectrum(sme)
    return sme


def adopt_bad_pixel_mask(sme: SME_Structure, mask: np.ndarray):
    # Set the mask, using only points that are close to the expected values
    print("Create bad pixel mask")
    sme.mask = sme.mask_values["bad"]
    # TODO determine threshold value
    threshold = 0.05
    for seg in range(sme.nseg):
        diff = np.abs(sme.spec[seg] - sme.synth[seg])
        close = diff < threshold
        upper = sme.spec[seg] > 0.95
        sme.mask[seg][close & upper] = sme.mask_values["continuum"]
        sme.mask[seg][close & (~upper)] = sme.mask_values["line"]

    fig = plot_plotly.FinalPlot(sme)
    fig.save(filename="mask.html")
    return sme


def fit_observation(
    sme: SME_Structure,
    star: Star,
    segments="all",
    parameters=["teff", "logg", "monh", "vsini", "vmac", "vmic"],
):
    # Fit the observation with SME
    print("Fit stellar spectrum with PySME")
    # sme.cscale_flag = "linear"
    # sme.cscale_type = "mask"
    # sme.vrad_flag = "whole"

    solver = SME_Solver()
    sme = solver.solve(sme, param_names=parameters, segments=segments)

    fig = plot_plotly.FinalPlot(sme)
    fig.save(filename="solved.html")

    # Save output
    print("Save results")
    for param in parameters:
        unit = getattr(star, param).unit
        print(f"{param}: {sme[param]} {unit}")
        setattr(star, param, sme[param] * unit)

    # TODO: + barycentric correction
    star.radial_velocity = sme.vrad[0] << (u.km / u.s)

    return sme, star


def first_guess(
    spectra: SpectrumArray, star: Star, blaze: np.ndarray, linelist: str, detector
):
    spectrum = combine_observations(spectra)
    spectrum = spectrum[0]
    sme = create_first_guess(
        spectrum, star, blaze, linelist, detector=detector
    )

    return sme


def extract_stellar_parameters(
    spectra: SpectrumArray, star: Star, blaze: np.ndarray, linelist: str
):
    sme = first_guess(spectra, star, blaze, linelist)
    sme = adopt_bad_pixel_mask(sme, None)
    sme, star = fit_observation(sme, star)
    return sme, star