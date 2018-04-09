"""
Test new stuff
"""
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.lines as mlines
from scipy.optimize import fsolve, minimize, minimize_scalar
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d as gaussbroad

from awlib.astro import doppler_shift, planck
from awlib.util import interpolate_DataFrame

#from test_project.Plot import Plot
from dataset import dataset
import intermediary as iy
import config
from stellar_db import stellar_db
from marcs import marcs
from harps import harps
from REDUCE import reduce
from limb_darkening import limb_darkening
from synthetic import synthetic
from idl import idl


def write(fname, obs):
    fname = join(conf['input_dir'], conf['dir_tmp'], fname)
    data = np.array([obs.wl, obs.flux[0], obs.err[0]]).swapaxes(0, 1)
    np.savetxt(fname, data, delimiter=', ')

def load(fname):
    fname = join(conf['input_dir'], conf['dir_tmp'], fname)
    wl, flx, err = np.genfromtxt(fname, delimiter=', ', unpack=True)
    ds = dataset(wl, flx, err)
    return ds

star = 'WASP-29'
planet = 'b'

conf = config.load_config(star + planet)
par = stellar_db.load_parameters(star, planet)

imu = np.geomspace(1, 0.0001, num=20)
imu[-1] = 0
conf['star_intensities'] = imu

def func(x):
    shift = doppler_shift(ds.wl, x)
    return -np.correlate(obs_flux, ds.__interpolate__(ds.wl, shift, ds.flux))[0]

sme = idl.load_stellar_flux(conf, par)
obs = reduce.load_stellar_flux(conf, par)

intensity = limb_darkening.load_specific_intensities(conf, par, obs)
telluric = dataset(obs.wl, np.ones_like(obs.flux))
syn = synthetic.load_observations(conf, par, telluric, obs, intensity)


plt.plot(sme.wl, sme.flux[0], label='SME syn')
plt.plot(obs.wl, obs.flux[0], label='observed')
plt.plot(syn.wl, syn.flux[0], label='syn observation')

plt.legend(loc='best')
plt.show()


fname = 'HARPS.2013-10-02T01:56:54.060c.ech'
fname = join(conf['input_dir'], conf['harps_dir'], fname)
obs = reduce.load(conf, par, fname)
#harps.flux_calibration(conf, par, obs, apply_temp_ratio=False, source='marcs', plot=True)

#shift = minimize(func, x0=23.5, method='Nelder-Mead')
#v = shift.x[0] + 1.5
v = -0.5
print('shift: ', v)
obs.doppler_shift(v)


tellurics = False
calib_dir = join(conf['input_dir'], 'HARPS', 'Calibration')
#solar = idl.load_solar(conf, par, calib_dir)
solar = idl.load_stellar_flux(conf, par)
#solar.gaussbroad(10)
solar.wl = obs.wl

ref = obs

###
# Create broadened profile
###

# Define Exclusion areas manually, usually telluric lines or detector gaps
# TODO get these areas automatically/from somewhere else
exclusion = np.array([(5880, 5910), (6340, 6410), (6562, 6565)])
tmp = np.zeros((exclusion.shape[0], obs.wl.shape[0]))
for i, ex in enumerate(exclusion):
    tmp[i] = ~((obs.wl > ex[0]) & (obs.wl < ex[1]))
tmp = np.all(tmp, axis=0)

# be careful to only broaden within individual sections
sensitivity = np.where(tmp, solar.flux / ref.flux, 0)

low, high = min(obs.wl), max(obs.wl)
for i in range(exclusion.shape[0] + 1):
    if i < exclusion.shape[0]:
        band = (obs.wl >= low) & (obs.wl < exclusion[i, 0])
        low = exclusion[i, 1]
    else:
        band = (obs.wl >= low) & (obs.wl < high)
    sensitivity[0, band] = gaussbroad(sensitivity[0, band], 500, mode='reflect')

sensitivity[0] = np.interp(obs.wl, obs.wl[tmp], sensitivity[0, tmp])

"""
bbflux = planck(obs.wl, 3800)  # Teff of the star
bbflux2 = planck(obs.wl, 5770)  # Teff of the sun
if apply_temp_ratio:
    # Fix Difference between solar and star temperatures
    ratio = bbflux2 / bbflux
else:
    ratio = 1
"""
ratio = 1
# Apply changes
calibrated = obs.flux * sensitivity * ratio
#calibrated[:, -50:] = calibrated[:, -51, None]
calibrated = calibrated[0]
calibrated = np.clip(calibrated, 0, 1)

x = obs.wl
plt.plot(obs.wl, obs.flux[0], label='observation')
plt.plot(solar.wl, solar.flux[0], label='solar')
plt.plot(x, calibrated, label='calibrated')
plt.plot(x, sensitivity[0], label='correction profile')
plt.legend(loc='best')

plt.show()

calibrated = dataset(x, calibrated[None, :], err = calibrated[None, :] * 0.001)
write('K23_calibrated_with_HD157881.asc', calibrated)

# Test harps flux calibrationc
reference = 'Vesta.fits'
ref = harps.load_solar(conf, par, reference=reference)
ref.doppler_shift(par['radial_velocity'])
#ref = harps.flux_calibration(conf, par, ref, apply_temp_ratio=False, plot=True, plot_title='Vesta')

#write('test.csv', r_wave, r_flux, r_err)

# Load HARPS
obs = harps.load_observations(conf, par)
bpmap = iy.create_bad_pixel_map(obs, threshold=1e-3)

obs.wl = obs.wl[~bpmap]

# Average HARPS flux
total = np.mean(obs.flux)
avg = np.mean(obs.flux, 1)
obs.scale *= total / avg[:, None]
obs.flux = np.mean(obs.flux, 0)

# Calibrate HARPS flux
obs.wl = obs.wl[obs.wl > 4000]
calib = harps.flux_calibration(conf, par, obs, plot=True, plot_title='K2-3')
calib.err = calib.flux * 0.001
write('harps.asc', calib)

# Load MARCS model
marc = marcs.load_stellar_flux(conf, par)
# TODO Factor 100???? Because Vesta reflected only 1% of the sunlight?
marc.wl = obs.wl * 0.01

fname = join(conf['input_dir'], conf['marcs_dir'], '3950g4.7z-0.25m0.6t0.flx')
marc2 = marcs.load_stellar_flux(conf, par, fname=fname)
# TODO Factor 100???? Because Vesta reflected only 1% of the sunlight?
marc2.wl = obs.wl * 0.01


# Load telluric spectrum
tell = harps.load_tellurics(conf, par)
tell.wl = obs.wl

marc.flux *= tell.flux
marc2.flux *= tell.flux
marc.gaussbroad(1)

bbflux = planck(obs.wl, 4000) / 100 / np.pi
bbflux2 = planck(obs.wl, 6770) / 100 / np.pi
ratio = bbflux2 / bbflux

#plt.plot(obs.wl, flux_harps, label='original')

plt.plot(obs.wl, marc.flux, label='marcs')
plt.plot(obs.wl, calib.flux[0], label='calibrated')
#plt.plot(obs.wl, marc2.flux, label='cold marcs')
#plt.plot(obs.wl, bbflux, label='4000K')
#plt.plot(obs.wl, bbflux2, label='6770K')
plt.legend(loc='best')
plt.show()
