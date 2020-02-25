"""
Create synthetic observation spectra, when telluric, stellar flux etc are given
"""

import logging
import numpy as np

from ..orbit import Orbit as orbit_calculator
from .data_interface import data_observations
from .dataset import dataset as dataset_classic
from .wavelengths import wavelengths


class dataset(dataset_classic):
    """Special dataset, that only evaluates the flux later"""

    def __init__(self, wave, data_func, err=None):
        super().__init__(wave, None, err=err)
        self._data_func = data_func

    @property
    def data(self):
        if self._data_orig is None:
            result = self._data_func()
            self._data_orig = self._data_func()
        else:
            result = self._data_orig
        result = self._broaden(result)
        result *= self.scale
        return result


class synthetic(wavelengths):
    """ create synthetic observation from given data """

    _obs_requires = ["parameters", "stellar_flux",
                     "intensities", "telluric", "planet"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wgrid = None
        self.flux = None
        self.parameters = None
        self.orbit = None
        self.phase = None
        self.time = None

    def get_observations(self, **data):
        obs = super().get_observations(**data)
        self.wgrid = obs.wave
        self.time = obs.time
        self.phase = obs.phase
        
        self.flux = self.synthetize
        self.parameters = data["parameters"]
        self.orbit = orbit_calculator(self.configuration, self.parameters)

        ds = dataset(self.wgrid, self.flux)
        ds.broadening = 1 / 2.355 * self.configuration['fwhm']
        ds.phase = self.phase
        ds.time = self.time
        return ds

    def synthetize(self):
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

        data = self._data_from_other_modules
        parameters = data["parameters"]
        stellar = data["stellar_flux"]
        telluric = data["tellurics"]
        i_core, i_atmo = data["intensities"]
        planet = data["planet"]

        phase = self.phase
        time = self.time

        # TODO interpolate all onto the same wavelength grid
        stellar.new_grid(self.wgrid)
        telluric.new_grid(self.wgrid)
        i_core.new_grid(self.wgrid)
        i_atmo.new_grid(self.wgrid)

        # Observed spectrum
        area_planet = parameters['A_planet+atm'].value
        area_atm = parameters['A_atm'].value
        obs = (stellar - i_core * area_planet +
               i_atmo * planet * area_atm) * telluric
        # Generate noise
        noise = np.random.randn(len(phase), len(
            self.wgrid)) / self.configuration['snr']

        # Apply instrumental broadening and noise
        obs *= (1 + noise)

        return obs.data
