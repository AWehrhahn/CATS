"""
Load data generated in IDL
"""
import glob
import os
from os.path import join
import logging

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as q

try:
    import pandexo.engine.justdoit as jdi

    hasPandexo = True
except ImportError:
    hasPandexo = False

from .data_interface import data_observations, data_stellarflux
from .dataset import dataset as dataset_classic


from ..orbit import Orbit

class dataset(dataset_classic):
    """Special dataset, that only evaluates the flux later"""

    def __init__(self, wave, data_func, err_func=None):
        super().__init__(wave, None, None)
        self._data_func = data_func
        self._err_func = err_func


    @property
    def data(self):
        if self._data_orig is None:
            self._data_orig = self._data_func()
        result = self._data_orig
        result = self._broaden(result)
        result = result * self.scale
        return result

    @property
    def error(self):
        if self._err_func is None:
            return None
        if self._err_orig is None:
            self._err_orig = self._err_func()
        result = self._err_orig
        result = self._broaden(result)
        result = result * self.scale
        return result

class pandexo(data_observations, data_stellarflux):

    _requires = ["parameters"]

    def __init__(self, configuration):
        super().__init__(configuration)
        self.instrument = configuration.get("instrument", "NIRSpec G395M")

    def simulate(self, star, planet, baseline=float("inf")):
        # Create inputs
        exo_dict = jdi.load_exo_dict()
        inst_dict = jdi.load_mode_dict(self.instrument)
        # Observation
        # saturation level in percent of full well
        exo_dict["observation"]["sat_level"] = 80
        exo_dict["observation"]["sat_unit"] = "%"
        # number of transits
        exo_dict["observation"]["noccultations"] = 10
        # fixed binning. I usually suggest ZERO binning.. you can always bin later
        # without having to redo the calcualtion
        exo_dict["observation"]["R"] = None
        # Defines how you specify out of transit observing time
        # 'frac' : fraction of time in transit versus out = in/out
        # 'total' : total observing time (seconds)
        exo_dict["observation"]["baseline_unit"] = "frac"
        # in accordance with what was specified above (total observing time)
        exo_dict["observation"]["baseline"] = 1
        # this can be a fixed level or it can be a filepath
        # to a wavelength dependent noise floor solution (units are ppm)
        exo_dict["observation"]["noise_floor"] = 0

        # Star
        # phoenix or user (if you have your own)
        exo_dict["star"]["type"] = "phoenix"
        # magnitude of the system
        exo_dict["star"]["mag"] = 8.0
        # For J mag = 1.25, H = 1.6, K =2.22.. etc (all in micron)
        exo_dict["star"]["ref_wave"] = 1.25
        exo_dict["star"]["temp"] = 5500  # in K
        exo_dict["star"]["metal"] = 0  # as log Fe/H
        exo_dict["star"]["logg"] = 4  # log surface gravity cgs

        # Planet
        # tells pandexo you are uploading your own spectrum
        exo_dict["planet"]["type"] = "user"
        exo_dict["planet"]["exopath"] = os.path.join(
            os.path.dirname(__file__), planet["exopath"]
        )
        # other options include "um","nm" ,"Angs", "sec" (for phase curves)
        exo_dict["planet"]["w_unit"] = "cm"
        # other options are 'fp/f*'
        exo_dict["planet"]["f_unit"] = "rp^2/r*^2"
        exo_dict["planet"]["transit_duration"] = 2.
        exo_dict["planet"]["td_unit"] = "hour"

        # TODO generate planet info from given parameters
        # tells pandexo you want a fixed transit depth
        # exo_dict["planet"]["type"] = "constant"
        # exo_dict["planet"]["transit_duration"] = 2.0
        # exo_dict["planet"]["td_unit"] = "h"
        # exo_dict["planet"]["radius"] = 1
        # # Any unit of distance in accordance with astropy.units can be added here
        # exo_dict["planet"]["r_unit"] = "R_jup"
        # exo_dict["star"]["radius"] = 1
        # exo_dict["star"]["r_unit"] = "R_sun"  # Same deal with astropy.units here
        # # this is what you would do for primary transit
        # exo_dict["planet"]["f_unit"] = "rp^2/r*^2"

        # Run simulation
        result = jdi.run_pandexo(exo_dict, inst_dict)
        return result

    def generate_observation(self):
        data = self._data_from_other_modules
        i_core, i_atmo = data["intensities"]
        i_core.new_grid(self.wave)
        depth = 1 - self.orbit.get_transit_depth(self.time)
        flux = self.flux[None, :] * (1 - i_core.data)
        return flux

    def generate_error(self):
        data = self._data_from_other_modules
        i_core, i_atmo = data["intensities"]
        i_core.new_grid(self.wave)
        depth = 1 - self.orbit.get_transit_depth(self.time)
        err = self.err[None, :] * (1 - i_core.data)
        return err

    def get_observations(self, **data):
        self.parameters = parameters = data["parameters"]
        self.orbit = Orbit(self.configuration, parameters)

        star = self.configuration["_star"]
        planet = self.configuration["_planet"]

        star = {
            "teff": parameters["teff"].to("K").value,
            "logg": parameters["logg"].to("").value,
            "monh": parameters["monh"].to("").value,
        }
        planet = {"exopath": "wasp12b.txt"}
        result = self.simulate(star, planet)
        self.wave = result["FinalSpectrum"]["wave"] * q.micrometer.to(q.Angstrom)
        self.flux = result["FinalSpectrum"]["spectrum_w_rand"]
        self.err = result["FinalSpectrum"]["error_w_floor"]

        n_obs = self.configuration["n_exposures"]
        period = self.parameters["period"].to("day").value
        t1 = self.orbit._backend.first_contact() - period / 100
        t4 = self.orbit._backend.fourth_contact() + period / 100
        self.time = np.linspace(t1, t4, n_obs)
        self.phase = self.orbit.get_phase(self.time)

        obs = dataset(self.wave, self.generate_observation, self.generate_error)
        obs.time = self.time
        obs.phase = self.phase
        return obs

    def get_stellarflux(self, **data):
        parameters = data["parameters"]
        star = {
            "teff": parameters["teff"].to("K").value,
            "logg": parameters["logg"].value,
            "monh": parameters["monh"].value,
        }
        planet = {"exopath": "wasp12b.txt"}
        # Simulate only out of transit observations, i.e. the stellar flux
        result = self.simulate(star, planet, baseline=0)
        wave = result["FinalSpectrum"]["wave"] * q.micrometer.to(q.Angstrom)
        flux = result["FinalSpectrum"]["spectrum_w_rand"]
        err = result["FinalSpectrum"]["error_w_floor"]
        sf = dataset_classic(wave, flux, err)
        return sf
