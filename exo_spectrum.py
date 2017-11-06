"""
Use an inverse method approach to determine the planetary transit spectrum
author: Ansgar Wehrhahn (ansgar.wehrhahn@physics.uu.se)
Based on work by Nikolai Piskunov (Uppsala University)
"""

import numpy as np

#from scipy.interpolate import interp1d
from scipy.linalg import solve_banded
from scipy.ndimage.filters import gaussian_filter1d as gaussbroad
import matplotlib.pyplot as plt

from read_write import read_write
#from solution import solution


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


def generate_spectrum(wl, kearth, flux, inten):
    """ Generate a fake spectrum """
    # Load planet spectrum
    planet = rw.load_input(wl)
    k_planet = gaussbroad(planet, sigma)
    x = np.arange(len(wl), dtype=np.float32)
    rand = np.random.rand(3, n_phase).astype(np.float32)

    # Amplitude
    ampl = rand[0] / snr
    # Period in pixels
    period = 500 + 1500 * rand[1]
    # Phase in radians
    phase = np.pi * rand[2]
    # Generate correlated noise
    err = np.cos(x[None, :] / period[:, None] * 2 * np.pi +
                 phase[:, None]) * ampl[:, None]
    # Observed spectrum
    s_p = (flux[None, :] - inten * sigma_p + inten *
           sigma_a * k_planet[None, :]) * kearth * (1 + err)
    # Generate noise
    noise1 = np.random.randn(n_phase, len(wl)) / snr

    obs = gaussbroad(s_p, sigma) * (1 + noise1)
    return obs


def solve(lamb):
    """
    Solve minimization problem for given lambda
    Obsolete
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

    # func = np.sum((so / ff - sigma_p / sigma_a * ke + ke *
    #               (np.tile(planet, n_phase).reshape((n_phase, len(planet)))) - obs / ff)**2)
    #reg = lamb * np.sum((sol[1:] - sol[:-1])**2)
    return sol  # , func, reg


def solve2(wl, f, g, lamb):
    """
    Solve minimization problem for given lambda
    """
    a = c = np.full(len(wl), - lamb, dtype=np.float32)

    b = np.sum(f, axis=0)
    r = np.sum(g, axis=0)
    b[:-1] += lamb
    b[1:] += lamb

    ab = np.array([a, b, c])
    return solve_banded((1, 1), ab, r)


# Step 1: Read data
rw = read_write(dtype=np.float32)
par = rw.load_parameters()
# Relative area of the atmosphere of the planet projected into the star
sigma_a = 2.6539e-6
# Relative area of the stelar disk covered by the planet
sigma_p = 8.3975e-5 + sigma_a
snr = 1e3                       # Signal to Noise Ratio
fwhm = par['fwhm']              # Instrumental FWHM in pixels
sigma = 1 / 2.355 * fwhm        # Sigma of Gaussian
n_phase = 200                   # Number of phases
nmu = 10
imu = np.around((np.arange(n_phase) / (n_phase - 1))
                * (nmu - 1) * 0.1 + 0.1, decimals=1)

# Load wavelength scale
s = rw.load_bin()
wl = s[0]
# obs = rw.load_observation('all')
flux, star_int = rw.load_star_model(
    wl, fwhm, 0, apply_normal=False, apply_broadening=False)
wl_tell, tell = rw.load_tellurics(wl, 1, apply_interp=False)
tell = tell[0]


# Doppler shift telluric spectrum
# Doppler shift
speed_of_light = 3e5  # km/s
velocity = np.linspace(0.5, n_phase + 0.5, num=n_phase, dtype=np.float32)
dop = 1 - velocity / speed_of_light
# Interpolate telluric spectrum
kearth = np.interp(wl[None, :] * dop[:, None], wl_tell, tell)
# Stellar spectrum blocked by planet / atmosphere
inten = star_int[imu].values.swapaxes(0, 1)  # TODO

# Generate fake spectrum
obs = generate_spectrum(wl, kearth, flux, inten)

# Broaden everything 
#gaussbroad = lambda x,y: x
ke = gaussbroad(kearth, sigma)
i_atm = gaussbroad(sigma_a * inten, sigma)
i_planet = gaussbroad(sigma_p * inten, sigma)
so = gaussbroad(flux[None, :] * kearth, sigma)


f = ke
g = obs / i_atm - so / i_atm + i_planet / i_atm * ke

# Find best lambda
lamb = 1000

sol2 = solve2(wl, f, g, lamb)

#sol, func, reg = solve(lamb)

"""
nlambda = 20
llambda = np.zeros(nlambda)
dev = np.zeros(nlambda)

for l in range(nlambda):
    lamb = lamb + 50
    sol, func, reg = solve(lamb)

    dev[l] = np.sum((sol - k_planet)**2)
    llambda[l] = lamb

    print('Lambda = ', lamb, ' Reconstruction = ',
          dev[l], ' Total = ', func + reg)


imin = np.argmin(dev)
lamb = llambda[imin]

# Recreate for plotting
sol, func, reg = solve(lamb)
"""

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
