"""
Test new stuff
"""
from os.path import join
import numpy as np
#import matplotlib.pyplot as plt
from scipy.optimize import fsolve

from awlib.astro import air2vac
from DataSources.PSG import PSG

from test_project.Plot import Plot

import config
import stellar_db
import marcs
import harps
import psg

plt = Plot()

star = 'K2-3'
planet = 'd'

conf = config.load_config(star + planet)
par = stellar_db.load_parameters(star, planet)

imu = np.geomspace(1, 0.0001, num=20)
imu[-1] = 0
conf['star_intensities'] = imu


wl_marcs, flux_marcs = marcs.load_flux_directly(conf, par)
wl_m2 , f_m2 = marcs.load_flux(conf, par)

f_m2 = np.interp(wl_marcs, wl_m2, f_m2)

#plt.plot(wl_marcs, flux_marcs/f_m2)
#plt.show()

#plt.plot(wl_marcs, flux_marcs, label='flux file')
#plt.plot(wl_marcs, f_m2, label='from spec. intensities')

#Load HARPS
fname = 'ADP.2015-01-23T22:55:24.657.fits'
fname = 'ADP.2016-03-04T01:02:58.223.fits'

wl_harps, flux_harps, phase = harps.load_observations(conf, par)

total = np.mean(flux_harps)
avg = np.mean(flux_harps, 1)

#TODO
flux_harps = flux_harps * total/avg[:, None]

wl_harps = wl_harps[0]
flux_harps = np.mean(flux_harps, 0)

# Get telluric
"""
psg_file = join(conf['input_dir'], conf['psg_dir'], conf['psg_file'])
_psg = PSG(config_file=psg_file)
tell_file = join(conf['input_dir'], conf['psg_dir'], conf['psg_file_tell'])
df = _psg.get_data_in_range(min(wl_marcs)/10000, max(wl_marcs)/10000, 1, wephm='T', type='tel')
df.to_csv(tell_file, index=False)
"""

wl_tell, tell = harps.load_tellurics(conf, par)

flux_marcs = np.interp(wl_harps, wl_marcs, flux_marcs)
tell = np.interp(wl_harps, wl_tell, tell)
#scaling
func = lambda x: np.sum((x * flux_harps - flux_marcs)**2)
x0 = max(flux_marcs) / max(flux_harps)
x = fsolve(func, x0)
flux_harps *= x[0]

plt.plot(wl_harps, flux_marcs * tell, name='marcs')
plt.plot(wl_harps, flux_harps, name='harps')
plt.plot(wl_harps, tell * np.max(flux_marcs), name='tellurics')
#plt.xlim([min(wl_harps), max(wl_harps)])

#plt.legend(loc='best')
plt.show()
