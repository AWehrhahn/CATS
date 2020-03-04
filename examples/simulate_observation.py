import logging
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.visualization import quantity_support
from specutils.spectra import SpectralRegion

from cats.data_modules import psg, sme, stellar_db, telluric_model
from cats.simulator.simulator import Simulator
from cats.simulator.detector import Crires

quantity_support()
#TODO: figure out logging (again)
logger = logging.getLogger(__name__)

data_directory = "/DATA/exoSpectro"
detector = Crires("H/1/4", 1)

# Define wavelength range
wrange = detector.regions
blaze = detector.blaze

# Load orbital and stellar parameters from Database
sdb = stellar_db.StellarDb()
star = sdb.get("GJ1214")
planet = star.planets["b"]

# Prepare stellar spectrum
stellar = sme.SmeStellar(star, linelist=f"{data_directory}/crires_h_1_4.lin")

# Prepare stellar intensities
intensities = sme.SmeIntensities(star, planet, linelist=f"{data_directory}/crires_h_1_4.lin")

# Prepare telluric spectrum
fname = f"{data_directory}/stdAtmos_crires_airmass1.fits"
telluric = telluric_model.TelluricModel(fname, star, "Cerro Paranal")

# Prepare planet spectrum
planet_spectrum = psg.PsgPlanetSpectrum(star, planet)

# Run Simulation
sim = Simulator(detector, star, planet, stellar, intensities, telluric, planet_spectrum)
spec = sim.simulate_series(wrange, 10)

# Compare to pure stellar spectrum
# Note: probably with different rest frame
star_spec = stellar.get(wave, 0)

plt.plot(spec.wavelength, spec.flux[0])
plt.plot(star_spec.wavelength, star_spec.flux)
plt.show()

pass
