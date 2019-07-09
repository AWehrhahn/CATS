import os.path
import numpy as np
import pandas as pd
import spectres

from ..orbit import orbit as orbit_calculator
from .dataset import dataset
from .data_interface import data_intensities, data_stellarflux

class marcs(data_intensities, data_stellarflux):
    def get_intensities(self, **data):
        obs = data["observations"]
        stellar = data["stellar_flux"]
        parameters = data["parameters"]

        orbit = orbit_calculator(self.configuration, parameters)

        mu = np.zeros(len(obs.data))
        for i in range(len(obs.data)):
            mu[i] = orbit.get_mu(*orbit.get_pos(orbit.get_phase(obs.time[i])))

        # intensity = stellar * limb_darkening(mu)

        return 1, 1

    def get_stellarflux(self, **data):

        flux_file = os.path.join(self.configuration['input_dir'],
                         self.configuration["marcs"]["dir"], "p3300_g+5.0_m0.0_t01_st_z+0.00_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.flx")
        wl_file = os.path.join(self.configuration['input_dir'],
                         self.configuration['marcs']['dir'], 'flx_wavelengths.vac')

        flux = pd.read_table(flux_file, header=None, names=['flx']).values.reshape(-1)
        wave = pd.read_table(wl_file, header=None, names=['wave']).values.reshape(-1)

        # wave = data["observations"][0].wave
        # flux = spectres.spectres(wave, wl, flux)
        flux /= 80

        ds = dataset(wave, flux)
        return ds