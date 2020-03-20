import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.time import Time
from astropy.constants import c
from astropy import units as u

from cats.simulator.detector import Crires
from cats.data_modules.stellar_db import StellarDb
from cats.reference_frame import PlanetFrame, TelescopeFrame, StarFrame
from exoorbit.orbit import Orbit

from scipy.optimize import minimize

from cats.solver.solution import __difference_matrix__, best_lambda, Tikhonov
from cats.solver.linear import LinearSolver


def standardize_spectrum(spectrum, wave, time, telescope_frame, planet_frame):
    rv = telescope_frame.to_frame(planet_frame, time)
    beta = rv / c
    shifted = np.copy(wave) * np.sqrt((1 + beta) / (1 - beta))
    spectrum[np.isnan(spectrum)] = 1
    spec = interp1d(shifted, spectrum, kind="linear", bounds_error=False)(wave)
    return spec


def nonlinear_leastsq(A, b, segment=5):
    def func(x, A, b):
        return A * x - b

    def reg(x, D):
        return D @ x

    def cost(x):
        cost = np.mean(func(x, A, b) ** 2)
        regul = regweight * np.mean(reg(x, D) ** 2)
        return cost + regul

    A = np.nan_to_num(np.asarray(A))
    b = np.nan_to_num(np.asarray(b))

    size = len(A[0])
    D = __difference_matrix__(size)
    regweight = best_lambda(np.mean(A, axis=0), np.mean(b, axis=0), plot=False)
    bounds = [(0, 1)] * size
    x0 = Tikhonov(np.mean(A, axis=0), np.mean(b, axis=0), regweight)
    x0 -= np.min(x0)
    x0 /= np.max(x0)

    res = minimize(
        cost,
        x0=x0,
        bounds=bounds,
        options={"maxiter": int(1e10), "maxfun": int(1e10), "iprint": 1},
    )
    return res.x


def gaussian_process(A, b, segment=5):
    import GPy

    X = [a[segment] for a in A]
    Y = [c[segment] for c in b]
    X = np.nan_to_num(np.asarray(X))
    Y = np.nan_to_num(np.asarray(Y))

    X = np.mean(X, axis=0)
    Y = np.mean(Y, axis=0)

    kernel = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=1)
    model = GPy.models.GPRegression(X, Y, kernel)
    model.optimize(messages=True)

    GPy.plotting.show(model.plot())

    return model


detector = Crires("H/1/4", [1, 2, 3])
sdb = StellarDb()
star = sdb.get("HD209458")
planet = star.planets["b"]
orbit = Orbit(star, planet)

transit_time = "2020-05-25T10:31:25.418"
transit_time = Time(transit_time, format="fits")
planet.time_of_transit = transit_time

wavelength = np.load("wave.npy")
times = np.load("times.npy")
img_spectra = np.load("spectra.npy")
img_telluric = np.load("telluric.npy")
img_stellar = np.load("stellar.npy")
img_intensities = np.load("intensities.npy")
planet_model = np.load("planet_model.npy")
times = Time(times, format="fits")

solver = LinearSolver(detector, star, planet)
wave, x0 = solver.solve(
    times,
    wavelength,
    img_spectra,
    img_stellar,
    img_intensities,
    img_telluric,
    regweight=200,
)

plt.plot(wavelength[32], planet_model)
plt.plot(wave, x0)
plt.show()
