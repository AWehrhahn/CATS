"""
Use an inverse method approach to determine the planetary transit spectrum
author: Ansgar Wehrhahn (ansgar.wehrhahn@physics.uu.se)
Based on work by Nikolai Piskunov (Uppsala University)
"""

import numpy as np

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
    #TODO fix dimensionality
    arr -= np.min(arr, axis=axis)[:, None]
    arr /= np.max(arr, axis=axis)[:, None]
    return arr

def normalize1d(arr):
    arr -= np.min(arr)
    arr /= np.max(arr)
    return arr    

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
    phase = np.pi * rand[2]
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
#TODO get values for sigma_a and sigma_p from orbital parameters
# Relative area of the atmosphere of the planet projected into the star
sigma_a = 2.6539e-6
# Relative area of the stelar disk covered by the planet
sigma_p = 8.3975e-5 + sigma_a
snr = 1e3                       # Signal to Noise Ratio
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
nmu = star_int.shape[1] - 1
imu = np.around(np.linspace(0.1, nmu * 0.1, n_phase), decimals=1)

#Load Tellurics
wl_tell, tell = rw.load_tellurics(wl, 60, apply_interp=False)
tell = tell[59]


print("Calculating intermediary data")
# Doppler shift telluric spectrum
# Doppler shift
speed_of_light = 3e5  # km/s
velocity = np.linspace(0.5, n_phase + 0.5, num=n_phase, dtype=np.float32)
dop = 1 - velocity / speed_of_light
# Interpolate telluric spectrum
tell = np.interp(wl[None, :] * dop[:, None], wl_tell, tell)
# Stellar spectrum blocked by planet / atmosphere
intensity = star_int[imu].values.swapaxes(0, 1)  # TODO

# Generate fake spectrum
obs_fake = generate_spectrum(wl, tell, flux, intensity)
obs = obs_fake

# Broaden everything
tell = gaussbroad(tell, sigma)
i_atm = gaussbroad(sigma_a * intensity, sigma)
i_planet = gaussbroad(sigma_p * intensity, sigma)
flux = gaussbroad(flux[None, :], sigma) #Add an extra dimension

f = tell
g = obs / i_atm - flux/ i_atm * tell + i_planet / i_atm * tell

print("Calculating solution")
# Find best lambda
lamb = 1000
sol2 = solution().solve(wl, f, g, lamb)

nn = len(wl) / 370
ww = rebin(wl, (nn,))
planet = rw.load_input(wl)
# Plot
#plt.plot(ww, rebin(sol, (nn,)), label='Best fit')
plt.plot(ww, rebin(sol2, (nn, )), label='Solution')
plt.plot(ww, rebin(planet, (nn, )), 'r', label='Input Spectrum')
plt.title('Lambda = %s, S//N = %s' % (lamb, snr))
plt.legend(loc='best')
plt.show()
