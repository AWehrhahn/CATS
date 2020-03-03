import logging
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.visualization import quantity_support
from specutils.spectra import SpectralRegion

from cats.data_modules import psg, sme, stellar_db, telluric_model
from cats.simulator.simulator import Simulator

quantity_support()
#TODO: figure out logging (again)
logger = logging.getLogger(__name__)

data_directory = "/DATA/exoSpectro"

wmin, wmax = 10000, 10050
wrange = SpectralRegion(wmin * u.AA, wmax * u.AA)
wave = np.linspace(wmin, wmax, 100) << u.AA

sdb = stellar_db.StellarDb()
star = sdb.get("GJ1214")
planet = star.planets["b"]

stellar = sme.SmeStellar(star, linelist=f"{data_directory}/sample.lin")
# plt.plot(stellar.wavelength, stellar.flux)
# plt.show()

intensities = sme.SmeIntensities(star, planet, linelist=f"{data_directory}/sample.lin")
# spec = intensities.get(wave, np.geomspace(0.01, 1, num=7))

# for i in range(7):
#     plt.plot(spec.wavelength, spec.flux[i])
# plt.show()

fname = f"{data_directory}/stdAtmos_crires_airmass1.fits"
telluric = telluric_model.TelluricModel(fname, star, "Cerro Paranal")
# tell = telluric.get(wave)

# plt.plot(tell.wavelength, tell.flux)
# plt.show()

planet_spectrum = psg.PsgPlanetSpectrum(star, planet)
# spec = planet_spectrum.get(wave)

# plt.plot(spec.wavelength, spec.flux)
# plt.show()

sim = Simulator(star, planet, stellar, intensities, telluric, planet_spectrum)
wave = sim.create_wavelength(wrange)
spec = sim.simulate_series(wave, 10)

star_spec = stellar.get(wave, 0)

plt.plot(spec.wavelength, spec.flux[0])
plt.plot(star_spec.wavelength, star_spec.flux)
plt.show()

pass
