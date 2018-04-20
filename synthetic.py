"""
Create synthetic observation spectra, when telluric, stellar flux etc are given
"""

import numpy as np
from scipy.ndimage.filters import gaussian_filter1d as gaussbroad

import orbit as orb
from data_module_interface import data_module
from psg import psg
from harps import harps
from idl import idl
from REDUCE import reduce

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
        cls.log(2, 'synthetic')
        cls.log(3, 'n_exposures: %i' % conf['n_exposures'])
        cls.log(3, 'snr: %i' % conf['snr'])
        cls.log(3, 'planet spectrum: %s' % source) 

        max_phase = np.pi - orb.maximum_phase(par)
        n_obs = conf['n_exposures']
        phase = np.random.uniform(low = np.pi - max_phase, high = np.pi + max_phase, size = n_obs)
        #phase = np.linspace(np.pi - max_phase, np.pi + max_phase, num=n_obs)
        #TODO do this properly, which restframe is that?
        tmp = stellar
        #stellar = reduce.load_stellar_flux(conf, par)

        # Sigma of Instrumental FWHM in pixels
        sigma = 1 / 2.355 * conf['fwhm']

        try:
            # Load planet spectrum
            if source == 'psg':
                planet = psg.load_input(conf, par)
                planet.wl = stellar.wl
                vel = orb.rv_planet(par, phase)
                planet.doppler_shift(vel)

        except FileNotFoundError:
            print('No planet spectrum for synthetic observation found')
            raise FileNotFoundError

        #planet = gaussbroad(planet, sigma)

        # Specific intensities
        i_planet, i_atm = orb.specific_intensities(par, phase, intensity)

        telluric.wl = stellar.wl
        i_planet.wl = stellar.wl
        i_atm.wl = stellar.wl

        # Observed spectrum
        area_planet = par['A_planet+atm']
        area_atm = par['A_atm']
        obs = (stellar - area_planet * i_planet + area_atm * i_atm * planet) * telluric
        # Generate noise
        noise = np.random.randn(len(phase), len(stellar.wl)) / conf['snr']

        # Apply instrumental broadening and noise
        obs.gaussbroad(sigma)
        obs.flux *= (1 + noise)
        obs.phase = phase
        return obs
