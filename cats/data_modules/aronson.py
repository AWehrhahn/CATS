"""
Fit limb darkening stellar intensities based in Erik Aronsons method

https://iopscience.iop.org/article/10.3847/1538-3881/aaa3fe/pdf

Assumptions:
1. Intensity is always positive
2. Brightness decreases towards the limb
3. The rate of brightness decrease is negative
4. Integral of all intensities is 1
"""
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.optimize import least_squares, curve_fit
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.linalg import svd

from spectres import spectres

from .data_interface import data_intensities
from ..orbit import Orbit as orbit_calculator
from ..solution import Tikhonov, best_lambda


class Intensity:
    def __init__(self, mu, steps):
        self.mu = mu
        self.steps = steps

    def __call__(self, mu):
        return self.calc_mu(mu)

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, value):
        if not np.all((value >= 0) & (value <= 1)):
            raise ValueError("Values of mu are expected to be between 0 and 1")
        if not np.all(np.diff(value) > 0):
            raise ValueError("Values of mu must be increasing")
        self._mu = value

    @property
    def steps(self):
        return self._steps

    @steps.setter
    def steps(self, value):
        if not np.all(value >= 0):
            raise ValueError("Steps must be positve")
        self._steps = value
        self._values = self.calc_step(np.argsort(self.mu)[::-1])

    @property
    def nsteps(self):
        return len(self.steps)

    def calc_step(self, n):
        n = np.asarray(n).ravel()
        tmp = np.zeros((n.size, self.nsteps), int)

        for i, ni in enumerate(n):
            tmp[i, :ni] = np.arange(1, ni + 1)[::-1]

        # tmp = np.arange(1, n+1)[::-1]
        return 1 - np.sum(tmp * self.steps, axis=1)

    def calc_mu(self, mu):
        # Use something other than linear interpolation?
        values = np.interp(mu, self.mu, self._values, left=0, right=0)
        # values = interp1d(self.mu, self._values, kind="cubic", fill_value=0, bounds_error=False)(mu)
        return values

    @staticmethod
    def fit(mu, intensity, x0=None, nsteps=101):

        # Remove invalid mu (e.g. from points outside the star)
        # Sort mu in increasing order
        mask = (mu <= 1) & (mu >= 0)
        mu, intensity = mu[mask], intensity[mask]
        sort = np.argsort(mu)
        mu, intensity = mu[sort], intensity[sort]

        # On an equispaced radius grid
        pillars = np.linspace(0, 1, nsteps)[::-1]
        pillars = np.sqrt(1 - pillars**2)
        if x0 is None:
            x0 = np.full(nsteps, 1 / nsteps**2)
            x0 += (np.random.rand(nsteps) - 0.5) * 1 / nsteps**2 * 1e-1

        obj = Intensity(pillars, x0)

        def func(steps):
            obj.steps = steps
            return obj.calc_mu(mu) - intensity

        res = least_squares(func, x0=x0, bounds=(0, 1))
        obj.steps = res.x
        return obj

    @classmethod
    def empty(cls, nsteps=101):
        pillars = np.linspace(0, 1, nsteps)[::-1]
        pillars = np.sqrt(1 - pillars**2)
        x0 = np.zeros(nsteps)
        obj = cls(pillars, x0)
        return obj

class aronson(data_intensities):
    def get_intensities(self, **data):
        # TODO get lightcurves from observations
        obs = data["observations"]
        self.parameters = parameters = data["parameters"]

        white = np.mean(obs.data, axis=1)
        In = 1 - (white-white.min()) / (white - white.min()).max()
        white /= np.percentile(white, 95)
        dates = obs.time

        self.orbit = orbit = orbit_calculator(self.configuration, parameters)
        phase = orbit.get_phase(dates)
        depths = orbit.get_transit_depth(dates)
        mu = orbit.get_mu(dates)
        radii = orbit.get_radius(dates)
        mu[mu == -1] = 0


        # Step 1: Fit first guess for limb darkening using the white light curve
        # Step 2: fit individual limb darkening for each wavelength


        # assert aronson.intensity(0, [0.1, 0.2, 0.1]) == 1
        # assert aronson.intensity(1, [0.1, 0.2, 0.1]) == 0.9

        # obj = self.fit2(dates, white)
        # datacube = np.copy(obs.data)
        # datacube /= np.percentile(datacube, 95, axis=0)
        # s = np.zeros((datacube.shape[1], len(obj.steps)))
        # v = np.zeros((datacube.shape[1], len(mu)))
        # r = np.zeros(datacube.shape[1])
        # e = np.zeros(datacube.shape[1])
        # for i in tqdm(range(datacube.shape[1])):
        #     tmp = self.fit2(dates, datacube[:, i])
        #     s[i] = tmp.steps
        #     v[i] = 1 - depths * tmp(mu)
        #     r[i] = tmp.r_planet
        #     e[i] = tmp.e_r_planet
        # np.save("steps.npy", s)
        # np.savez("radii.npy", radius=r, wave=obs.wave, error=e)
        # np.savez("phase_curve.npy", curve=v, phase=phase)

        s = np.load("steps.npy")
        tmp = np.load("radii.npy.npz")
        r = tmp["radius"][:, 0]
        e = tmp["error"][:, 0]
        wave = tmp["wave"]
        tmp = np.load("phase_curve.npy.npz")
        v = tmp["curve"]
        phase = tmp["phase"]

        f = 1 / e**2
        g = r / e**2
        # Normalize f and g
        a = f.mean()
        f /= a
        g /= a
        lamb = best_lambda(f, g, ratio=50, spacing=wave)
        sol = Tikhonov(f, g, lamb, spacing=wave)
        np.save("radius_smooth.npy", np.ma.filled(sol))

        planet = data["planet"]
        p_wave = planet.wave
        p_data = planet.data * 200 + 17400
        p_data = spectres(wave[10:-10], p_wave, p_data)
        p_wave = wave[10:-10]

        plt.plot(p_wave, p_data, label="Input")
        plt.plot(wave, r, label="Radius")
        plt.plot(wave, sol, label="Smoothed")
        plt.legend()
        plt.show()
        pass

    def fit(self, dates, white, x0=None):
        mu = self.orbit.get_mu(dates)
        mu[mu < 0] = 0
        depths = self.orbit.get_transit_depth(dates)
        # white = white / white.max()
        obj = Intensity.empty()

        def func(steps):
            obj.steps = steps
            lk = 1 - depths * obj(mu)
            return lk - white

        if x0 is None:
            x0 = obj.steps
        res = least_squares(func, x0=x0, bounds=(0, 1))
        obj.steps = res.x
        return obj

    def fit2(self, dates, flux, x0=None):
        r0 = self.parameters["r_planet"].to("km").value
        r_star = self.parameters["r_star"].to("km").value

        # Those don't depend on the planet radius
        phase = self.orbit.get_phase(dates)
        mu = self.orbit.get_mu(dates)
        mu[mu < 0] = 0

        def func(r):
            self.orbit._backend.planet.radius = r[0]
            depths = self.orbit.get_transit_depth(dates)
            obj = self.fit(dates, flux, x0=x0)
            lc = 1 - depths * obj(mu)
            weight = np.sqrt(depths * (r_star / r[0])**2)
            return (flux - lc) * weight

        res = least_squares(func, x0=(r0,), xtol=1e-12, gtol=None, ftol=None, diff_step=1e-4)
        self.orbit._backend.planet.radius = res.x[0]

        obj = self.fit(dates, flux)
        obj.r_planet = res.x[0]

        _, s, VT = svd(res.jac, full_matrices=False)
        pcov = np.dot(VT.T / s**2, VT)
        obj.e_r_planet = np.sqrt(pcov[0, 0])

        return obj



    @staticmethod
    def intensity(n, steps):
        # I_n = I_0 - sum_1^n((n-i+1) s_n)
        n = np.asarray(n).ravel()
        nn = n.size
        tmp = np.zeros((nn, len(steps)), int)
        steps = np.abs(steps)

        for i in range(nn):
            tmp[i, :n[i]] = np.arange(1, n[i] + 1)[::-1]

        # tmp = np.arange(1, n+1)[::-1]
        return 1 - np.sum(tmp * steps, axis=1)

    @staticmethod
    def normalization(steps, radii):
        dr = np.gradient(radii)
        return 2 * np.sum(aronson.intensity(np.arange(len(steps)), steps) * dr)

    @staticmethod
    def normalized_intensity(n, steps, radii):
        return aronson.intensity(n, steps) / aronson.normalization(steps, radii)
