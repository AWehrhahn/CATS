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

from scipy.optimize import least_squares

from .data_interface import data_intensities
from ..orbit import orbit as orbit_calculator

class aronson(data_intensities):
    def get_intensities(self, **data):
        # TODO get lightcurves from observations
        obs = data["observations"]
        parameters = data["parameters"]

        white = obs.data.sum(axis=1)
        dates = obs.time

        orbit = orbit_calculator(self.configuration, parameters)
        phase = orbit.get_phase(dates)

        # plt.plot(phase, white)
        # plt.show()

        # Step 1: Fit first guess for limb darkening using the white light curve
        # Step 2: fit individual limb darkening for each wavelength


        nsteps = 100
        radii = np.linspace(0, 1, nsteps)

        # TODO limb darkening law
        mu = np.sqrt(1- radii**2)
        gamma1 = 0.1
        gamma2 = 0.2
        In = 1 - gamma1 * (1 - mu) - gamma2 * (1-mu)**2

        n = np.arange(nsteps)
        res = least_squares(lambda x: aronson.normalized_intensity(n, x, radii) - In, x0=np.zeros(nsteps), bounds=(0, np.inf))

        steps = res.x

        plt.plot(radii, In)
        plt.plot(radii, aronson.normalized_intensity(n, steps, radii))
        plt.show()

        norm = aronson.normalization(steps, radii)

        pass

    @staticmethod
    def intensity(n, steps):
        # I_n = I_0 - sum_1^n((n-i+1) s_n)
        nn = np.size(n)
        tmp = np.zeros((nn, len(steps)), int)

        for i in range(nn):
            tmp[i, :n[i]] = np.arange(1, n[i]+1)[::-1]

        # tmp = np.arange(1, n+1)[::-1]
        return 1 - np.sum(tmp * steps, axis=0) 

    @staticmethod
    def normalization(steps, radii):
        return 2 * np.sum(aronson.intensity(np.arange(len(steps)), steps) * radii)

    @staticmethod
    def normalized_intensity(n, steps, radii):
        return aronson.intensity(n, steps) / aronson.normalization(steps, radii)
