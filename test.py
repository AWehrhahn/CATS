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

def write(fname, wl, flux, err):
    fname = join(conf['input_dir'], conf['dir_tmp'], fname)
    data = np.array([wl, flux, err]).swapaxes(0, 1)
    np.savetxt(fname, data, delimiter=', ')


star = 'K2-3'
planet = 'd'

conf = config.load_config(star + planet)
par = stellar_db.load_parameters(star, planet)

imu = np.geomspace(1, 0.0001, num=20)
imu[-1] = 0
conf['star_intensities'] = imu

"""
#Test harps flux calibrationc
reference = 'Vesta.fits'
ref = harps.load_solar(conf, par, reference=reference)
ref.doppler_shift(par['radial_velocity'])
#r_wave = doppler_shift(r_wave, par['radial_velocity'])
ref = harps.flux_calibration(conf, par, r_wave, r_flux, r_err, tellurics=False, plot=True, plot_title='Vesta')

r_flux = r_flux[0]
write('test.csv', r_wave, r_flux, r_err)
"""
#Load HARPS
wl_harps, flux_harps, phase = harps.load_observations(conf, par)
bpmap = iy.create_bad_pixel_map(flux_harps, threshold=1e-3)

flux_harps = flux_harps[:, ~bpmap]
wl_harps = wl_harps[~bpmap]

#Average HARPS flux
total = np.mean(flux_harps)
avg = np.mean(flux_harps, 1)
flux_harps = flux_harps * total/avg[:, None]
wl_harps = wl_harps
flux_harps = np.mean(flux_harps, 0)

#Calibrate HARPS flux
flux_harps = flux_harps[wl_harps > 4000]
wl_harps = wl_harps[wl_harps > 4000]
err_harps = np.full_like(wl_harps, 0.002)

flux_calib = harps.flux_calibration(conf, par, wl_harps, flux_harps, err_harps , plot=True)[0][0]
write('harps.asc', wl_harps, flux_calib, np.full_like(flux_calib, 0.002))

#Load MARCS model
wl_marcs, flux_marcs = marcs.load_stellar_flux(conf, par)
flux_marcs = np.interp(wl_harps, wl_marcs, flux_marcs) * 0.01 #TODO Factor 100???? Because Vesta reflected only 1% of the sunlight?

fname = join(conf['input_dir'], conf['marcs_dir'], '3950g4.7z-0.25m0.6t0.flx')
wl_marcs2, flux_marcs2 = marcs.load_stellar_flux(conf, par, fname=fname)
flux_marcs2 = np.interp(wl_harps, wl_marcs2, flux_marcs2) * 0.01 #TODO Factor 100???? Because Vesta reflected only 1% of the sunlight?


#Load telluric spectrum
wl_tell, flux_tell = harps.load_tellurics(conf, par)
flux_tell = interp1d(wl_tell, flux_tell, kind='quadratic',
                    bounds_error=False)(wl_harps)

flux_marcs *= flux_tell
flux_marcs2 *= flux_tell
flux_marcs = gaussbroad(flux_marcs, 1)
#flux_calib = gaussbroad(flux_calib, 1)

bbflux = planck(wl_harps, 4000) / 100 / np.pi
bbflux2 = planck(wl_harps, 6770) / 100 / np.pi
ratio = bbflux2 / bbflux

#plt.plot(wl_harps, flux_harps, label='original')

plt.plot(wl_harps, flux_marcs, label='marcs')
plt.plot(wl_harps, flux_calib * ratio, label='calibrated')
#plt.plot(wl_harps, flux_marcs2, label='cold marcs')
#plt.plot(wl_harps, bbflux, label='4000K')
#plt.plot(wl_harps, bbflux2, label='6770K')
plt.legend(loc='best')
plt.show()

"""
wl_i, factors = marcs.load_limb_darkening(conf, par)
wl_m2 , f_m2 = marcs.load_flux(conf, par)
f_m2 = np.interp(wl_marcs, wl_m2, f_m2)

plt.plot(wl_marcs, flux_marcs - f_m2, label='directly')
#plt.plot(wl_m2, f_m2, label='intensities')
plt.legend(loc='best')
plt.show()

f_m2 = np.interp(wl_marcs, wl_m2, f_m2)
wl_m3, intensities = marcs.load_intensities(conf, par)
"""

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
