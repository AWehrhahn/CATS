"""
Get Data from Stellar DB
"""

import logging
import numpy as np

from functools import lru_cache

# from scipy import constants as const
from astropy import constants as const
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord

from exoorbit.bodies import Body, Star, Planet
from data_sources.StellarDB import StellarDB as SDB

from .datasource import DataSource

logger = logging.getLogger(__name__)


class StellarDb(DataSource):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = DataSource.__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        super().__init__()
        self.backend = SDB()

    @lru_cache(128)
    def get(self, name):
        """Load the data on the star from the local database, or online
        if not available locally.
        
        Parameters
        ----------
        name : str
            Name of the star / planet
        
        Returns
        -------
        star : exoorbit.bodies.Star
            recovered Star
        """

        data = self.backend.load(name)

        # Convert names
        # Stellar parameters
        coords = SkyCoord(data["coordinates"]["ra"], data["coordinates"]["dec"])
        star = Star(
            name=name,
            mass=data["mass"],
            radius=data["radius"],
            effective_temperature=data["t_eff"],
            logg=data["logg"],
            monh=data["metallicity"],
            vturb=data["velocity_turbulence"],
            coordinates=coords,
            distance=data["distance"],
            radial_velocity=data["radial_velocity"],
        )

        planets = {}
        for pname, p in data["planets"].items():
            planet = Planet(
                name=pname,
                radius=p["radius"],
                mass=p["mass"],
                inclination=p["inclination"],
                semi_major_axis=p["semi_major_axis"],
                period=p["period"],
                eccentricity=p["eccentricity"],
                # argument_of_periastron=Time(p["periastron"],format="jd"),
                time_of_transit=Time(p["transit_epoch"], format="jd"),
                transit_duration=p["transit_duration"],
            )

            planet.teff = (
                (np.pi * planet.radius ** 2) / planet.a ** 2
            ) ** 0.25 * star.teff

            if planet.mass > 10 * u.M_earth:
                # Hydrogen (e.g. for gas giants)
                planet.atm_molar_mass = 2.5 * (u.g / u.mol)
            else:
                # dry air (mostly nitrogen)  (e.g. for terrestial planets)
                planet.atm_molar_mass = 29 * (u.g / u.mol)

            # assuming isothermal atmosphere, which is good enough on earth usually
            planet.atm_scale_height = (
                const.R
                * planet.teff
                * planet.radius ** 2
                / (const.G * planet.mass * planet.atm_molar_mass)
            )

            planets[pname] = planet

        star.planets = planets

        return star
