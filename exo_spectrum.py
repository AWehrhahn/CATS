"""
Use an inverse method approach to determine the planetary transit spectrum
author: Ansgar Wehrhahn (ansgar.wehrhahn@physics.uu.se)
Based on work by Nikolai Piskunov (Uppsala University)
"""

import numpy as np

from scipy.interpolate import interp1d
from scipy.linalg import solve_banded
from scipy.ndimage.filters import gaussian_filter1d as gaussbroad
import matplotlib.pyplot as plt

from read_write import read_write


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


def solve(lamb):
    """
    Solve minimization problem for given lambda
    """
    lam = np.full(len(wl), lamb)

    a = np.zeros(len(wl))
    c = np.zeros(len(wl))
    b = np.zeros(len(wl))
    r = np.zeros(len(wl))

    a[:-1] = -lam[:-1]
    c[1:] = -lam[1:]

    for iphase in range(n_phase):
        g = obs[iphase, :] / ff[iphase, :] - so[iphase, :] / \
            ff[iphase, :] + sigma_p / sigma_a * ke[iphase, :]
        b = b + ke[iphase, :]
        r = r + g

    b[:-1] = b[:-1] + lam[:-1]
    b[1:] = b[1:] + lam[1:]

    ab = np.array([a, b, c])
    sol = solve_banded((1, 1), ab, r)

    func = np.sum((so / ff - sigma_p / sigma_a * ke + ke *
                   (np.tile(planet, n_phase).reshape((n_phase, len(planet)))) - obs / ff)**2)
    reg = lamb * np.sum((sol[1:] - sol[:-1])**2)
    return sol, func, reg


# Step 1: Read data
rw = read_write()
par = rw.load_parameters()
# Relative area of the atmosphere of the planet projected into the star
sigma_a = 2.6539e-6
# Relative area of the stelar disk covered by the planet
sigma_p = 8.3975e-5 + sigma_a
snr = 1e3                       # Signal to Noise Ratio
fwhm = par['fwhm']              # Instrumental FWHM in pixels
sigma = 1 / 2.355 * fwhm          # Sigma of Gaussian
n_phase = 200                   # Number of phases
nmu = 10
imu = np.around((np.arange(n_phase) / (n_phase - 1))
                * (nmu - 1) * 0.1 + 0.1, decimals=1)

_, wl = rw.load_observation(1)
planet = rw.load_input(wl)
flux, star_int = rw.load_star_model(
    wl, fwhm, 0, apply_normal=False, apply_broadening=False)
wl_tell, tell = rw.load_tellurics(wl, 1, apply_interp=False)
tell = tell[0]

# Generate fake spectrum
k_planet = gaussbroad(planet, sigma)
obs = np.zeros((n_phase, len(wl)))  # observation
ff = np.zeros((n_phase, len(wl)))   # star_data
so = np.zeros((n_phase, len(wl)))   # stellar_flux
ke = np.zeros((n_phase, len(wl)))   # telluric

x = np.arange(len(wl))
for iphase in range(n_phase):
    phase = np.random.rand(30)
    ampl = phase[9] / snr * 0             # Amplitude
    period = 500 + 1500 * phase[19]       # Period in pixels
    phase = 2 * np.pi * phase[29]         # Phase in radians
    err = np.cos(x / period * 4 * np.pi + phase) * \
        ampl   # Generate correlated noise
    dop = 1 - (1 * iphase / (n_phase - 1) + 0.5) / 3e5          # Doppler shift
    kearth = interp1d(wl_tell, tell, fill_value='extrapolate')(wl * dop)
    inten = star_int[imu[iphase]].values
    s_p = (flux - inten * sigma_p + inten *
           sigma_a * k_planet) * kearth * (1 + err)
    noise1 = np.random.randn(len(wl)) / snr

    obs[iphase, :] = gaussbroad(s_p, sigma)
    obs[iphase, :] *= 1 + noise1
    ke[iphase, :] = gaussbroad(kearth, sigma)
    ff[iphase, :] = gaussbroad(sigma_a * inten, sigma)
    so[iphase, :] = gaussbroad(flux * kearth, sigma)

# Find best lambda
lamb = 9e10
nlambda = 20
llambda = np.zeros(nlambda)
dev = np.zeros(nlambda)

for l in range(nlambda):
    lamb = lamb + 1e7
    sol, func, reg = solve(lamb)

    dev[l] = np.sum((sol - k_planet)**2)
    llambda[l] = lamb

    print('Lambda = ', lamb, ' Reconstruction = ',
          dev[l], ' Total = ', func + reg)


imin = np.argmin(dev)
lamb = llambda[imin]

# Recreate for plotting
sol, func, reg = solve(lamb)

nn = len(wl) / 370
ww = rebin(wl, (nn,))

# Plot
plt.plot(ww, rebin(k_planet, (nn,)), label='Best fit')
plt.plot(ww, rebin(planet, (nn, )), 'r', label='Input Spectrum')
plt.title('Lambda = %s, S//N = %s' % (lamb, snr))
plt.legend(loc='best')
plt.show()
