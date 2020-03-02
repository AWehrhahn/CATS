import numpy as np
import astropy.units as u
from astropy.visualization import quantity_support
import matplotlib.pyplot as plt

from cats.spectrum import Spectrum1D

quantity_support()

n = 1000
wave = np.linspace(5600, 6800, n) << u.AA
flux = np.random.randn(n) << u.Unit(1)

s = Spectrum1D(spectral_axis=wave, flux=flux)

wave2 = np.linspace(5600, 6800, n // 3) << u.AA

s = s.resample(wave2)


plt.plot(s.wavelength, s.flux)
plt.show()
