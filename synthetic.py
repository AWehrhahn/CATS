"""
Create synthetic observation spectra, when telluric, stellar flux etc are given
"""

import numpy as np
from scipy.ndimage.filters import gaussian_filter1d as gaussbroad

import intermediary as iy
from data_module_interface import data_module
from psg import psg

from awlib.timeit import timeit


class synthetic(data_module):
    """ create synthetic observation from given data """

    @classmethod
    @timeit
    def load_observations(cls, conf, par, telluric, stellar, intensity, source='psg'):
        """ Generate a fake spectrum """

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