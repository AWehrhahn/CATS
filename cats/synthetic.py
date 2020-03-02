"""
Create synthetic observation spectra, when telluric, stellar flux etc are given
"""

import logging
import numpy as np

from .orbit import Orbit as orbit_calculator
from .data_modules.datasource import DataSource


class Synthetic(DataSource):
    """ create synthetic observation from given data """

    def __init__(self, parameters):
        super().__init__()
        self.parameters = parameters
        self.orbit = orbit_calculator(self.config, self.parameters)

        # Use evenly spaced time points between first and fourth contact
        self.n_obs = self.config["n_exposures"]
        self.wmin = self.config["wavelength_minimum"]
        self.wmax = self.config["wavelength_maximum"]
        self.snr = self.config["snr"]
        self.R = self.config["resolution"]

        # Load wavelength grid definition
        # Use geomspace for even sampling in frequency space

    @staticmethod
    def get_number_of_wavelengths_points_from_resolution(R, wmin, wmax):
        def gen(R, wmin, wmax):
            delta_wave = lambda w: w / R
            wave_local = wmin
            yield wave_local
            while wave_local < wmax:
                wave_local += delta_wave(wave_local)
                yield wave_local
            return

        generator = gen(R, wmin, wmax)
        ls = list(generator)
        return len(ls)

    def synthetize(self, stellar, telluric, i_core, i_atmo, planet):
        """ Generates a synthetic spectrum based on the input spectra

        A planetary transmission spectrum is taken from the module defined with source
        Observations are generated over the whole transit
        Noise is added, to achive the SNR defined in conf

        Parameters:
        ----------
        ** data : dict
            previously loaded and calculated data

        Returns
        -------
        obs : dataset
            synthetic observations
        """

        parameters = self.parameters

        # Calculate phase
        period = self.parameters["period"].to("day").value
        t1 = self.orbit._backend.first_contact() - period / 100
        t4 = self.orbit._backend.fourth_contact() + period / 100
        time = np.linspace(t1, t4, self.n_obs)
        phase = self.orbit.get_phase(time)

        # Create new wavelength grid
        wpoints = Synthetic.get_number_of_wavelengths_points_from_resolution(
            self.R, self.wmin, self.wmax
        )
        wgrid = np.geomspace(self.wmin, self.wmax, wpoints)
        wgrid[0] = self.wmin
        wgrid[-1] = self.wmax

        # interpolate all onto the same wavelength grid
        method = "flux_conserved"
        planet = planet.resample(wgrid, method=method)
        stellar = stellar.resample(wgrid, method=method)
        telluric = telluric.resample(wgrid, method=method)
        i_core = i_core.resample(wgrid, method=method)
        i_atmo = i_atmo.resample(wgrid, method=method)

        # Observed spectrum
        area_planet = parameters["A_planet+atm"].value
        area_atm = parameters["A_atm"].value
        obs = (stellar - i_core * area_planet + i_atmo * planet * area_atm) * telluric
        # Generate noise
        noise = np.random.randn(len(phase), len(wgrid)) / self.config["snr"]

        # Apply instrumental broadening and noise
        obs *= 1 + noise

        return obs
