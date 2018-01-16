"""
Test new stuff
"""
from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

from awlib.astro import air2vac, doppler_shift
from awlib.util import interpolate_DataFrame
from DataSources.PSG import PSG

#from test_project.Plot import Plot
import intermediary as iy
import config
import stellar_db
import marcs
import harps
import psg

#plt = Plot()

star = 'K2-3'
planet = 'd'

conf = config.load_config(star + planet)
par = stellar_db.load_parameters(star, planet)

imu = np.geomspace(1, 0.0001, num=20)
imu[-1] = 0
conf['star_intensities'] = imu

"""
wl_i, factors = marcs.load_limb_darkening(conf, par)
"""
wl_marcs, flux_marcs = marcs.load_flux_directly(conf, par)
"""
wl_m2 , f_m2 = marcs.load_flux(conf, par)

f_m2 = np.interp(wl_marcs, wl_m2, f_m2)


plt.plot(wl_marcs, flux_marcs - f_m2, label='directly')
#plt.plot(wl_m2, f_m2, label='intensities')
plt.legend(loc='best')
plt.show()

f_m2 = np.interp(wl_marcs, wl_m2, f_m2)
wl_m3, intensities = marcs.load_intensities(conf, par)
"""
#Load HARPS
wl_harps, flux_harps, phase = harps.load_observations(conf, par)
bpmap = iy.create_bad_pixel_map(flux_harps, threshold=1e-3)
bpmap[wl_harps > 6800] = True

flux_harps = flux_harps[:, ~bpmap]
wl_harps = wl_harps[~bpmap]

#flux_harps[0] = doppler_shift(wl_harps, flux_harps[0], -par['radial_velocity'])

#Average HARPS flux
total = np.mean(flux_harps)
avg = np.mean(flux_harps, 1)
flux_harps = flux_harps * total/avg[:, None]
wl_harps = wl_harps
flux_harps = np.mean(flux_harps, 0)

flux_calib = harps.flux_calibration(conf, par, wl_harps, flux_harps)[0]

plt.plot(wl_marcs, flux_marcs, label='marcs')
plt.plot(wl_harps, flux_harps * 100, label='original')
plt.plot(wl_harps, flux_calib * 100, label='calibrated')
plt.legend(loc='best')
plt.show()

# Get telluric
wl_tell, tell = harps.load_tellurics(conf, par)

factors = interpolate_DataFrame(wl_harps, wl_i, factors)
intensities = interpolate_DataFrame(wl_harps, wl_m3, intensities)

flux_marcs = np.interp(wl_harps, wl_marcs, flux_marcs)
tell = np.interp(wl_harps, wl_tell, tell)
#scaling
func = lambda x: np.sum((x * flux_harps - flux_marcs)**2)
x0 = max(flux_marcs) / max(flux_harps)
x = fsolve(func, x0)
flux_harps *= x[0]

#Combine with MARCS for specific intensities
si = factors.apply(lambda s: s * flux_harps)

#plt.plot(wl_m3, flux_marcs, label='flux')
plt.plot(wl_harps, intensities[1], label='intensity[0]')
plt.plot(wl_harps, si[1], label='combined')
plt.xlim([min(wl_harps), max(wl_harps)])

plt.legend(loc='best')
plt.show()

plt.plot(wl_harps, flux_marcs * tell, label='marcs')
plt.plot(wl_harps, flux_harps, label='harps')
plt.plot(wl_harps, tell * np.max(flux_marcs), label='tellurics')
plt.xlim([min(wl_harps), max(wl_harps)])

plt.legend(loc='best')
plt.show()
