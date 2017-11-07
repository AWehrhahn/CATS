"""
Use an inverse method approach to determine the planetary transit spectrum
author: Ansgar Wehrhahn (ansgar.wehrhahn@physics.uu.se)
Based on work by Nikolai Piskunov (Uppsala University)
"""

import numpy as np
import os.path

from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d as gaussbroad
import matplotlib.pyplot as plt

from read_write import read_write
from solution import solution


def rebin(a, newshape):
    '''
    Rebin an array to a new shape.
    '''
    assert len(a.shape) == len(newshape)

    slices = [slice(0, old, float(old) / new)
              for old, new in zip(a.shape, newshape)]
    coordinates = np.mgrid[slices]
    # choose the biggest smaller integer index
    indices = coordinates.astype('i')
    return a[tuple(indices)]


def normalize2d(arr, axis=1):
    """ normalize array arr """
    # TODO fix dimensionality
    arr -= np.min(arr, axis=axis)[:, None]
    arr /= np.max(arr, axis=axis)[:, None]
    return arr


def normalize1d(arr):
    """ normalize array arr """
    arr -= np.min(arr)
    arr /= np.max(arr)
    return arr


def interpolate_intensity(mu, i):
    """ interpolate the stellar intensity for given limb distance mu """
    return interp1d(i.keys().values, i.values, kind=rw.config['interpolation_method'], fill_value='extrapolate', bounds_error=False)(mu).swapaxes(0, 1)


def calc_mu(phase):
    """ calculate the distance from the center of the planet to the center of the star as seen from earth """
    return par['sma'] / par['r_star'] * \
        np.sqrt(np.cos(par['inc'])**2 +
                np.sin(par['inc'])**2 * np.sin(phase)**2)


def generate_spectrum(wl, telluric, flux, intensity):
    """ Generate a fake spectrum """
    # Load planet spectrum
    planet = rw.load_input(wl)
    planet = gaussbroad(planet, sigma)
    x = np.arange(len(wl), dtype=np.float32)
    rand = np.random.rand(3, n_phase).astype(np.float32)

    # Amplitude
    amplitude = rand[0] / snr
    # Period in pixels
    period = 500 + 1500 * rand[1]
    # Phase in radians
    phase_max = par['r_star'] / par['sma']
    phase = np.linspace(-phase_max, phase_max, n_phase)
    mu = calc_mu(phase)
    intensity = interpolate_intensity(mu, intensity)
    #phase = np.pi * rand[2]
    # Generate correlated noise
    error = np.cos(x[None, :] / period[:, None] * 2 * np.pi +
                   phase[:, None]) * amplitude[:, None]
    # Observed spectrum
    obs = (flux[None, :] - intensity * sigma_p + intensity *
           sigma_a * planet[None, :]) * telluric * (1 + error)
    # Generate noise
    noise = np.random.randn(n_phase, len(wl)) / snr

    obs = gaussbroad(obs, sigma) * (1 + noise)
    return obs


# Step 1: Read data
print("Loading data")
rw = read_write(dtype=np.float32)
par = rw.load_parameters()

# Relative area of the stelar disk covered by the planet and atmosphere
sigma_p = ((par['h_atm'] + par['r_planet']) / par['r_star'])**2
# Relative area of the atmosphere of the planet projected into the star
sigma_a = sigma_p - (par['r_planet'] / par['r_star'])**2

snr = par['snr']                       # Signal to Noise Ratio
fwhm = par['fwhm']              # Instrumental FWHM in pixels
sigma = 1 / 2.355 * fwhm        # Sigma of Gaussian


# Load wavelength scale
"""
s = rw.load_bin()
wl = s[0]
n_phase = 200           # Number of observations
"""
obs, wl = rw.load_observation('all')
n_phase = obs.shape[0]

# Load stellar model
flux, star_int = rw.load_star_model(
    wl, fwhm, 0, apply_normal=False, apply_broadening=False)
nmu = len(star_int)
imu = np.around(np.linspace(0.1, nmu * 0.1, n_phase), decimals=1)

# Load Tellurics
wl_tell, tell = rw.load_tellurics(wl, n_phase // 2 + 1, apply_interp=False)
tell = tell[n_phase // 2]  # central tellurics, has no rv shift ?


print("Calculating intermediary data")
# Doppler shift telluric spectrum
# Doppler shift
speed_of_light = 3e5  # km/s
velocity = np.linspace(0.5, n_phase + 0.5, num=n_phase, dtype=np.float32)
dop = 1 - velocity / speed_of_light
# Interpolate telluric spectrum
tell = np.interp(wl[None, :] * dop[:, None], wl_tell, tell)
# Stellar spectrum blocked by planet / atmosphere
# intensity = {i: star_int[i].swapaxes(0, 1) for i in star_int.keys()}  # TODO

# Generate fake spectrum
obs = normalize2d(obs * star_int[0.0][None, :])
obs_fake = generate_spectrum(wl, tell, flux, star_int)

# TODO extract phase from observation
phase_max = par['r_star'] / par['sma']
phase = np.linspace(-phase_max, phase_max, n_phase)
mu = calc_mu(phase)
intensity = interpolate_intensity(mu, star_int)

#test = -np.max(intensity * sigma_p, axis=1)
#obs = test[:, None] * obs
obs = obs_fake


# Broaden everything
tell = gaussbroad(tell, sigma)
i_atm = gaussbroad(sigma_a * intensity, sigma)
i_planet = gaussbroad(sigma_p * intensity, sigma)
flux = gaussbroad(flux[None, :], sigma)  # Add an extra dimension

f = tell
g = obs / i_atm - flux / i_atm * tell + i_planet / i_atm * tell

print("Calculating solution")
# Find best lambda
lamb = 1e6
sol2 = solution().solve(wl, f, g, lamb)
#sol2 = normalize1d(sol2)

planet = rw.load_input(wl)
# Plot
#plt.plot(ww, rebin(sol, (nn,)), label='Best fit')
plt.plot(wl, planet, 'r', label='Input Spectrum')
plt.plot(wl, sol2, label='Solution')
plt.title('Lambda = %s, S//N = %s' % (lamb, snr))
plt.legend(loc='best')

# save plot
output_file = os.path.join(rw.output_dir, rw.config['file_spectrum'])
plt.savefig(output_file, bbox_inches='tight')
# save data
output_file = os.path.join(rw.output_dir, rw.config['file_data_out'])
np.savetxt(output_file, sol2)

plt.show()
