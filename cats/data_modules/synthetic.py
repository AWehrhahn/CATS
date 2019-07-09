"""
Create synthetic observation spectra, when telluric, stellar flux etc are given
"""

import logging
import numpy as np

from ..orbit import orbit as orbit_calculator
from .data_interface import data_observations



class synthetic(data_observations):
    """ create synthetic observation from given data """

    _requires = ["parameters", "stellar_flux", "intensities", "telluric", "planet"]

    def get_observations(self, **data):
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
        logging.info('synthetic')
        logging.info('n_exposures: %i', self.configuration['n_exposures'])
        logging.info('snr: %i', self.configuration['snr'])
        logging.info('planet spectrum: %s', self.configuration["source"])

        parameters = data["parameters"]
        stellar = data["stellar_flux"]
        telluric = data["telluric"]
        i_core, i_atmo = data["intensities"]

        # TODO
        planet = data["planet"]

        orbit = orbit_calculator(self.configuration, parameters)
        max_phase = np.pi - orbit.maximum_phase()
        n_obs = self.configuration['n_exposures']
        phase = np.random.uniform(low=np.pi - max_phase, high=np.pi + max_phase, size=n_obs)

        # Sigma of Instrumental FWHM in pixels
        sigma = 1 / 2.355 * self.configuration['fwhm']

        # TODO interpolate all onto the same wavelength grid
        telluric.wave = stellar.wave
        i_core.wave = stellar.wave
        i_atmo.wave = stellar.wave

        # Observed spectrum
        area_planet = parameters['A_planet+atm']
        area_atm = parameters['A_atm']
        obs = (stellar - area_planet * i_core + area_atm * i_atmo * planet) * telluric
        # Generate noise
        noise = np.random.randn(len(phase), len(stellar.wl)) / self.configuration['snr']

        # Apply instrumental broadening and noise
        obs.gaussbroad(sigma)
        obs.flux *= (1 + noise)
        obs.phase = phase
        # TODO get times instead of phase
        return obs
