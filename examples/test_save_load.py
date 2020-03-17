from os.path import join, dirname
from cats.data_modules.sme import SmeStellar
from cats.data_modules.stellar_db import StellarDb
from cats.simulator.detector import Crires

from cats.spectrum import SpectrumList
from cats.reference_frame import TelescopeFrame

from astropy.time import Time
import matplotlib.pyplot as plt
import numpy as np


detector = Crires("H/1/4", [1, 2, 3])
observatory = detector.observatory
wrange = detector.regions

transit_time = "2020-05-25T10:31:25.418"
transit_time = Time(transit_time, format="fits")

sdb = StellarDb()
star = sdb.get("HD209458")
planet = star.planets["b"]
data_dir = join(dirname(__file__), "noise_zero")

frame = TelescopeFrame(transit_time, detector.observatory, star.coordinates)

fname = "test.fits"

stellar = SmeStellar(star, linelist=f"{data_dir}/crires_h_1_4.lin")
spec = stellar.get(wrange, transit_time)
spec = spec.shift(frame)

spec.write(fname)

spec2 = SpectrumList.read(fname)

plt.plot(
    np.concatenate(spec.wavelength).value,
    np.concatenate(spec.flux).decompose().value,
    label="pre-save",
)
plt.plot(
    np.concatenate(spec2.wavelength).value,
    np.concatenate(spec2.flux).value,
    "--",
    label="post-save",
)
plt.legend()
plt.show()
