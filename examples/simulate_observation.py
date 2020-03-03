import numpy as np
import astropy.units as u
from specutils.spectra import SpectralRegion

from cats.data_modules import psg, sme, stellar_db
from cats.simulator.simulator import Simulator

data_directory = "/DATA/exoSpectro"

wrange = SpectralRegion(5000 * u.AA, 60000 * u.AA)

wave = np.linspace(5000, 6000, 10000) << u.AA

sdb = stellar_db.StellarDb()
star = sdb.get("GJ1214")
planet = star.planets["b"]

stellar = sme.SmeStellar(star, linelist=f"{data_directory}/4000-6920.lin")

spec = stellar.get(wave)

intensities = sme.SmeIntensities(star, planet)
telluric = None
planet_spectrum = psg.PsgPlanetSpectrum(star, planet)

Simulator(star, planet, stellar, intensities, telluric, planet_spectrum)
