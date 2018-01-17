"""
Create synthetic observation spectra, when telluric, stellar flux etc are given
"""

import numpy as np
from scipy.ndimage.filters import gaussian_filter1d as gaussbroad

import intermediary as iy
from data_module_interface import data_module
from psg import psg


class synthetic(data_module):
    """ create synthetic observation from given data """

    @classmethod
    def load_observations(cls, conf, par, wl_tell, telluric, wl_flux, flux, intensity, source='psg'):
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
                wl, planet = psg.load_input(conf, par, None)
        except FileNotFoundError:
            print('No planet spectrum for synthetic observation found')
            raise FileNotFoundError

        #planet = gaussbroad(planet, sigma)

        # Specific intensities
        _i_planet, _i_atm = iy.specific_intensities(par, phase, intensity)

        flux = np.interp(wl, wl_flux, flux)
        telluric = np.interp(wl, wl_tell, telluric)

        i_planet = np.zeros((len(phase), len(wl)))
        i_atm = np.zeros((len(phase), len(wl)))

        for i in range(len(phase)):
            i_planet[i] = np.interp(wl, wl_flux, _i_planet[i])
            i_atm[i] = np.interp(wl, wl_flux, _i_atm[i])

        # Observed spectrum
        obs = (flux[None, :] - i_planet * par['A_planet+atm'] +
               par['A_atm'] * i_atm * planet[None, :]) * telluric
        # Generate noise
        noise = np.random.randn(len(phase), len(wl)) / conf['snr']

        # Apply instrumental broadening and noise
        obs = gaussbroad(obs, sigma) * (1 + noise)
        return wl, obs, phase
