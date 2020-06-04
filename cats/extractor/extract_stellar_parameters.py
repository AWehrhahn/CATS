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

from ..data_modules.stellar_db import StellarDb
from ..simulator.detector import Crires
from ..spectrum import SpectrumArray


def round_to_nearest(value, options):
    value = np.atleast_2d(value).T
    options = np.asarray(options)

    diff = np.abs(value - options)
    sort = np.argsort(diff)
    nearest = options[sort[:, 0]]
    if value.size == 1:
        return nearest[0]
    return nearest


def combine_observations(spectra, blaze):
    # TODO: The telluric spectrum will change between observations
    # and therefore influence the recovered stellar parameters
    # Especially when we combine data from different transits!

    # Shift to the same reference frame (barycentric)
    print("Shift observations to the barycentric restframe")
    spectra = spectra.shift("barycentric", inplace=True)

    # Arbitrarily choose the central grid as the common one
    print("Combine all observations")
    wavelength = spectra.wavelength[len(spectra) // 2]
    spectra = spectra.resample(wavelength)
    spectrum = np.nansum(spectra.flux, axis=0)
    spectrum = SpectrumArray(
        flux=spectrum[None, :],
        spectral_axis=wavelength[None, :],
        segments=spectra.segments,
        datetime=[spectra.datetime[len(spectra) // 2]],
    )

    # Normalize to upper envelope
    print("Normalize combined spectrum")
    spectrum.flux /= blaze.ravel()
    for left, right in zip(spectrum.segments[:-1], spectrum.segments[1:]):
        spectrum.flux[left:right] /= np.nanpercentile(spectrum.flux[left:right], 95)

    spectrum = spectrum[0]
    return spectrum


def create_first_guess(spectrum, star, blaze, linelist):
    print("Extracting stellar parameters...")

    # Create SME structure
    print("Preparing PySME structure")
    sme = SME_Structure()
    sme.wave = [wave.to_value(u.AA) for wave in spectrum.wavelength]
    sme.spec = [spec.to_value(1) for spec in spectrum.flux]

    sme.teff = star.teff.to_value(u.K)
    sme.logg = star.logg.to_value(1)
    sme.monh = star.monh.to_value(1)
    sme.vturb = star.vturb.to_value(u.km / u.s)

    sme.abund = "solar"
    sme.linelist = ValdFile(linelist)

    vturb = round_to_nearest(sme.vturb, [1, 2, 5])
    atmosphere = f"marcs2012s_t{vturb:1.1f}.sav"
    sme.atmo.source = atmosphere
    sme.atmo.method = "grid"

    nlte = None
    if nlte is not None:
        for elem, grid in nlte.items():
            sme.nlte.set_nlte(elem, grid)

    sme.cscale_flag = "none"
    sme.normalize_by_continuum = True
    sme.vrad_flag = "fix"
    sme.vrad = star.radial_velocity.to_value("km/s")

    # Create an initial spectrum using the nominal values
    # This also determines the radial velocity
    print("Determine the radial velocity using the nominal stellar parameters")
    synthesizer = Synthesizer()
    sme = synthesizer.synthesize_spectrum(sme)
    return sme


def adopt_bad_pixel_mask(sme, mask):
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


def fit_observation(sme, star, segments="all"):
    # Fit the observation with SME
    print("Fit stellar spectrum with PySME")
    sme.cscale_flag = "linear"
    sme.cscale_type = "mask"
    sme.vrad_flag = "whole"

    solver = SME_Solver()
    sme = solver.solve(sme, param_names=["teff", "logg", "monh"], segments=segments)

    fig = plot_plotly.FinalPlot(sme)
    fig.save(filename="solved.html")

    # Save output
    print("Save results")
    print(f"Teff: {sme.teff} K")
    print(f"logg: {sme.logg} log(cm/s**2)")
    print(f"MonH: {sme.monh} dex")

    star.effective_temperature = sme.teff * u.K
    star.logg = sme.logg * u.one
    star.monh = sme.monh * u.one
    return sme, star


def extract_stellar_parameters(spectra, star, blaze, linelist):
    spectrum = combine_observations(spectra, blaze)
    sme = create_first_guess(spectrum, star, blaze, linelist)
    sme = adopt_bad_pixel_mask(sme, None)
    sme, star = fit_observation(sme, star)
    return sme, star
