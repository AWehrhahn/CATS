import numpy as np
from astropy.time import Time
import matplotlib.pyplot as plt

from cats.data_modules.psg import PsgPlanetSpectrum
from cats.data_modules.stellar_db import StellarDb
from cats.simulator.detector import Crires
from cats.spectrum import SpectrumList

stellar = SpectrumList.read("stellar.fits")
wave = stellar.wavelength

detector = Crires("H/1/4", [1, 2, 3])
wrange = detector.regions

sdb = StellarDb()
star = sdb.get("HD209458")
planet = star.planets["b"]

psg = PsgPlanetSpectrum(star, planet)
spec = psg.get(wrange, Time.now())
spec = spec.resample(wave)
spec = detector.apply_instrumental_broadening(spec)

spec = np.concatenate(spec.flux).to_value(1)
wave = np.concatenate(wave).to_value("AA")
np.save("planet_model.npy", spec)

plt.plot(spec)
plt.show()

