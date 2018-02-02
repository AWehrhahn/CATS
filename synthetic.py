"""
Create synthetic observation spectra, when telluric, stellar flux etc are given
"""

import numpy as np
from scipy.ndimage.filters import gaussian_filter1d as gaussbroad

import intermediary as iy
from data_module_interface import data_module
from psg import psg

import matplotlib.pyplot as plt


class synthetic(data_module):
    """ create synthetic observation from given data """

    @classmethod
    def load_observations(cls, conf, par, telluric, stellar, intensity, source='psg', *args, **kwargs):
        """ Generates a synthetic spectrum based on the input spectra

        A planetary transmission spectrum is taken from the module defined with source
        Observations are generated over the whole transit
        Noise is added, to achive the SNR defined in conf

        Parameters:
        ----------
        conf : {dict}
            configuration setting
        par : {dict}
            stellar and planetary parameters
        telluric : {dataset}
            telluric transmission spectrum
        stellar : {dataset}
            stellar flux
        intensity : {dataset}
            specific intensities for various mu values
        Raises
        ------
        FileNotFoundError
            Planetary spectrum file not found

        Returns
        -------
        obs : dataset
            synthetic observations
        """
        # TODO determine suitable phases independently
        max_phase = iy.maximum_phase(par)
        n_obs = 20
        phase = np.linspace(np.pi - max_phase, np.pi + max_phase, num=n_obs)

        # Sigma of Instrumental FWHM in pixels
        sigma = 1 / 2.355 * conf['fwhm']

        try:
            # Load planet spectrum
            if source == 'psg':
                planet = psg.load_input(conf, par)
                planet.wl = stellar.wl
        except FileNotFoundError:
            print('No planet spectrum for synthetic observation found')
            raise FileNotFoundError

        #planet = gaussbroad(planet, sigma)

        # Specific intensities
        i_planet, i_atm = iy.specific_intensities(par, phase, intensity)

        telluric.wl = stellar.wl
        i_planet.wl = stellar.wl
        i_atm.wl = stellar.wl

        # Observed spectrum
        obs = (stellar - i_planet * par['A_planet+atm'] +
               par['A_atm'] * i_atm * planet) * telluric
        # Generate noise
        noise = np.random.randn(len(phase), len(stellar.wl)) / conf['snr']

        # Apply instrumental broadening and noise
        obs.flux = gaussbroad(obs.flux, sigma) * (1 + noise)
        obs.phase = phase
        return obs
