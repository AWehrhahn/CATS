import logging
import numpy as np

from ..orbit import Orbit as orbit_calculator
from .data_interface import data_observations
from .dataset import dataset

class wavelengths(data_observations):

    @staticmethod
    def get_number_of_wavelengths_points_from_resolution(R, wmin, wmax):
        def gen(R, wmin, wmax):
            delta_wave = lambda w: w/R
            wave_local = wmin
            yield wave_local
            while wave_local < wmax:
                wave_local += delta_wave(wave_local)
                yield wave_local
            return

        generator = gen(R, wmin, wmax)
        ls = list(generator)
        return len(ls)


    def get_observations(self, **data):
        cls = self.__class__
        # Load wavelength grid definition
        # Use geomspace for even sampling in frequency space
        wmin = self.configuration["wavelength_minimum"]
        wmax = self.configuration["wavelength_maximum"]
        R = self.configuration["resolution"]
        wpoints = cls.get_number_of_wavelengths_points_from_resolution(R, wmin, wmax)

        wgrid = np.geomspace(wmin, wmax, wpoints)
        wgrid[0] = wmin
        wgrid[-1] = wmax
        flux = None
        obs = dataset(wgrid, flux)

        if "parameters" in data.keys():
            parameters = data["parameters"]
            orbit = orbit_calculator(self.configuration, parameters)
            n_obs = self.configuration['n_exposures']
            t1 = orbit._backend.first_contact()  - parameters["period"].to("day").value / 100
            t4 = orbit._backend.fourth_contact() + parameters["period"].to("day").value / 100
            time = np.linspace(t1, t4, n_obs)
            phase = orbit.get_phase(time)
            obs.time = time
            obs.phase = phase
        return obs
