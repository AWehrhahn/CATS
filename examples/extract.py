import numpy as np
from os.path import dirname, join
from glob import glob

from astropy import units as u
import astroplan as ap
from astropy.coordinates import SkyCoord

from cats.spectrum import SpectrumList
from cats.simulator.detector import Crires
from cats.data_modules.stellar_db import StellarDb
from cats.data_modules.sme import SmeIntensities

from exoorbit import Orbit

from pysme.sme import SME_Structure
from pysme.solve import SME_Solver
from pysme.linelist.vald import ValdFile

data_dir = join(dirname(__file__), "transit")
files = join(data_dir, "b_*.fits")

detector = Crires("H/1/4", [1, 2, 3])

spectra = [SpectrumList.read(f) for f in glob(files)]
times = [spec[0].datetime for spec in spectra]
wrange = detector.regions
# Star and Planet nominal data
star = spectra[0][0]["star"]
planet = spectra[0][0]["planet"]

# Sort spectra by time
sort = np.argsort(times)

# Correct for blaze function
blaze = detector.blaze
spectra /= blaze

# TODO Continuum normalize

# Add all spectra together in barycentric frame
# and then fit sme to get accurate stellar parameters
spectra_star = spectra.shift("star")
stellar = sum(spectra)

sme = SME_Structure()
sme.wave = stellar.wavelength.to_value(u.AA)
sme.spec = stellar.flux.to_value(u.one)

linelist=f"{data_dir}/crires_h_1_4.lin"
sme.linelist = ValdFile(linelist)

solver = SME_Solver()
sme = solver.solve(sme, ["teff", "logg", "monh"])

# TODO Compare to nominal values for the star
star.teff = sme.teff << u.K
star.logg = sme.logg
star.monh = sme.monh

# Use the obs times to get phases and calculate specific intensities for those
si = SmeIntensities(star, planet, linelist=linelist)
intensities = si.get(wrange, times)

# Get telluric spectra (from observations?)
# Shift everything to telescope restframe
# Fit airmass versus spectrum, to get telluric

spectra_telescope = spectra.shift("telescope")

observer = ap.Observer.at_site(detector.observatory)
coords = SkyCoord(star.ra, star.dec)
target = ap.FixedTarget(name=star.name, coord=coords)
altaz = observer.altaz(times, target)
airmass = altaz.secz.value

coeff = np.polyfit(airmass, spectra_telescope, 1)

# Compare to expectation
telluric_at_one = np.polyval(coeff, 1)

# Solve equation
