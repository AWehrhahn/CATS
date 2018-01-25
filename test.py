"""
Test new stuff
"""
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, minimize
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d as gaussbroad

from awlib.astro import doppler_shift, planck
from awlib.util import interpolate_DataFrame

#from test_project.Plot import Plot
import intermediary as iy
import config
import stellar_db
from marcs import marcs
from harps import harps
from synthetic import synthetic

def write(fname, obs):
    fname = join(conf['input_dir'], conf['dir_tmp'], fname)
    data = np.array([obs.wl, obs.flux, obs.err]).swapaxes(0, 1)
    np.savetxt(fname, data, delimiter=', ')


star = 'K2-3'
planet = 'd'

conf = config.load_config(star + planet)
par = stellar_db.load_parameters(star, planet)

imu = np.geomspace(1, 0.0001, num=20)
imu[-1] = 0
conf['star_intensities'] = imu


#Test harps flux calibrationc
reference = 'Vesta.fits'
ref = harps.load_solar(conf, par, reference=reference)
ref.doppler_shift(par['radial_velocity'])
#r_wave = doppler_shift(r_wave, par['radial_velocity'])
ref = harps.flux_calibration(conf, par, ref, apply_temp_ratio=False, plot=True, plot_title='Vesta')

#write('test.csv', r_wave, r_flux, r_err)

#Load HARPS
obs = harps.load_observations(conf, par)
bpmap = iy.create_bad_pixel_map(obs.flux, threshold=1e-3)

obs = obs[~bpmap]
#obs.wl = obs.wl[~bpmap]

#Average HARPS flux
total = np.mean(obs.flux)
avg = np.mean(obs.flux, 1)
obs.scale *= total/avg[:, None]
obs.flux = np.mean(obs.flux, 0)

#Calibrate HARPS flux
obs.wl = obs.wl[obs.wl > 4000]
calib = harps.flux_calibration(conf, par, obs , plot=True, plot_title='K2-3')
write('harps.asc', calib)

#Load MARCS model
marc = marcs.load_stellar_flux(conf, par)
marc.wl = obs.wl * 0.01 #TODO Factor 100???? Because Vesta reflected only 1% of the sunlight?

fname = join(conf['input_dir'], conf['marcs_dir'], '3950g4.7z-0.25m0.6t0.flx')
marc2 = marcs.load_stellar_flux(conf, par, fname=fname)
marc2.wl = obs.wl * 0.01 #TODO Factor 100???? Because Vesta reflected only 1% of the sunlight?


#Load telluric spectrum
tell = harps.load_tellurics(conf, par)
tell.wl = obs.wl

marc.flux *= tell.flux
marc2.flux *= tell.flux
marc.flux = gaussbroad(marc.flux, 1)
#flux_calib = gaussbroad(flux_calib, 1)

bbflux = planck(obs.wl, 4000) / 100 / np.pi
bbflux2 = planck(obs.wl, 6770) / 100 / np.pi
ratio = bbflux2 / bbflux

#plt.plot(obs.wl, flux_harps, label='original')

plt.plot(obs.wl, marc.flux, label='marcs')
plt.plot(obs.wl, calib.flux * ratio, label='calibrated')
#plt.plot(obs.wl, marc2.flux, label='cold marcs')
#plt.plot(obs.wl, bbflux, label='4000K')
#plt.plot(obs.wl, bbflux2, label='6770K')
plt.legend(loc='best')
plt.show()
