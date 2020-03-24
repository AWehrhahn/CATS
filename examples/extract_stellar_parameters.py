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

from cats.data_modules.sme import SmeIntensities, SmeStellar
from cats.data_modules.stellar_db import StellarDb
from cats.data_modules.telluric_model import TelluricModel
from cats.data_modules.psg import PsgPlanetSpectrum
from cats.simulator.detector import Crires
from cats.spectrum import SpectrumList
from cats import reference_frame as rf
from cats.reference_frame import PlanetFrame, TelescopeFrame
from exoorbit import Orbit, Star


def round_to_nearest(value, options):
    value = np.atleast_2d(value).T
    options = np.asarray(options)

    diff = np.abs(value - options)
    sort = np.argsort(diff)
    nearest = options[sort[:, 0]]
    if value.size == 1:
        return nearest[0]
    return nearest


data_dir = join(dirname(__file__), "noise_1")
files = join(data_dir, "*.fits")

linelist = f"{data_dir}/crires_h_1_4.lin"

detector = Crires("H/1/4", [1, 2, 3])
observatory = detector.observatory
wrange = detector.regions
blaze = detector.blaze

# Load the nominal values for an initial guess
sdb = StellarDb()
star = sdb.get("HD209458")

# Load data
spectra = [SpectrumList.read(f) for f in tqdm(glob(files))]
times = Time([spec.datetime for spec in tqdm(spectra)])

# Sort by observation time
sort = np.argsort(times)
spectra = [spectra[i] for i in sort]
times = times[sort]

# Shift to the same reference frame (barycentric)
spectra = [spec.shift("barycentric", inplace=True) for spec in tqdm(spectra)]

# Option A:
#   - resample all spectra to a common wavelength grid
#   - add them together
#   - fit to that
# Pros:
#   - easier to normalize
#   - radial velocity shift between observations is small (if from the same transit)
# Cons:
#   - Loose some precision due to resampling

# Option B:
#   - calculate SME spectra without sampling to wavelength grid
#   - do least-squares to each spectrum individually and add them together
# Pros:
#   - more precise, since no interpolation
# Cons:
#   - difficult to normalize individual spectra
#   - more effort

# Lets do Option A!

# Arbitrarily choose the central grid as the common one
wavelength = spectra[len(spectra) // 2].wavelength
spectra = [spec.resample(wavelength) for spec in spectra]

# Add everything together
spectrum = spectra[0].flux
for i in range(1, len(spectra)):
    spectrum = [spec1 + spec2 for spec1, spec2 in zip(spectrum, spectra[i].flux)]

spectrum = [spec.to_value(u.one) for spec in spectrum]
# Normalize to upper envelope
spectrum = [spec / b.to_value(1) for spec, b in zip(spectrum, blaze)]
spectrum = [spec / np.nanpercentile(spec, 95) for spec in spectrum]

# Create SME structure
sme = SME_Structure()
sme.wave = [wave.to_value(u.AA) for wave in wavelength]
sme.spec = spectrum

sme.teff = star.teff.to_value(u.K)
sme.logg = star.logg
sme.monh = star.monh
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
sme.vrad_flag = "whole"
sme.vrad = star.radial_velocity.to_value("km/s")

# Create an initial spectrum using the nominal values
# This also determines the radial velocity
synthesizer = Synthesizer()
sme = synthesizer.synthesize_spectrum(sme)

# Set the mask, using only points that are close to the expected values
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

# Fit the observation with SME
sme.cscale_flag = "constant"
sme.cscale_type = "mask"
sme.vrad_flag = "fix"

solver = SME_Solver()
sme = solver.solve(sme, param_names=["teff", "logg", "monh"])

fig = plot_plotly.FinalPlot(sme)
fig.save(filename="solved.html")

# Save output
star.effective_temperature = sme.teff * u.K
star.logg = sme.logg * u.one
star.monh = sme.monh * u.one

star.save("star.yaml")
pass
