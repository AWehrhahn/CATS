"""
Get Data from Stellar DB
"""

import logging
import numpy as np
# from scipy import constants as const
from astropy import constants as const
from astropy import units as q
from astropy.time import Time

from .data_interface import data_orbitparameters

# TODO rework stellar_db
from data_sources.StellarDB import StellarDB


class stellar_db(data_orbitparameters):
    def get_parameters(self, **_):
        """ loads the stellar parameters from the StellarDB database

        Converts measurements into km, s, and radians and calculates some simple fixed parameters like the projected area of the planet on the star
        See StellarDB project for more details
        """
        # name of the star, if not found in local StellarDB it will automatically be resolved with SIMBAD
        name_star = self.configuration["_star"]
        # name of the planet, just the letter, e.g. 'b'
        name_planet = self.configuration["_planet"]

        sdb = StellarDB()
        star = sdb.load(name_star)
        # Convert names
        # Stellar parameters
        star['name_star'] = star['name'][0]
        star['r_star'] = (star['radius'] * q.R_sun).to(q.km)
        star['m_star'] = (star['mass'] * q.M_sun).to(q.kg)
        star['teff'] = star['t_eff'] * q.Kelvin
        star['logg'] = star['logg'] * q.one
        star['monh'] = star['metallicity'] * q.one
        star['star_vt'] = star['vel_turb'] * q.km / q.s
        # Planetary parameters
        planet = star['planets'][name_planet]
        star['name_planet'] = name_planet
        star['r_planet'] = (planet['radius'] * q.R_jupiter).to(q.km)
        star['m_planet'] = (planet['mass'] * q.M_jupiter).to(q.kg)
        star['inc'] = planet['inclination'] * q.deg
        if 'atm_height' in planet.keys():
            star['h_atm'] = planet['atm_height'] * q.km

        star['sma'] = (planet['semi_major_axis'] * q.AU).to(q.km)
        star['period'] = planet['period'] * q.day
        star['transit'] = Time(planet['transit_epoch'], format="jd")
        star['duration'] = planet['transit_duration'] * q.day 
        star['ecc'] = planet['eccentricity'] * q.one
        # TODO
        star["w"] = 90 * q.deg

        logging.info('T_eff: %s, logg: %.2f, [M/H]: %.1f' % (star['teff'], star['logg'], star['metallicity']))

        # TODO: atmosphere model
        # stellar flux in = thermal flux out
        star['T_planet'] = ((np.pi * star['r_planet']**2) /
                            star['sma']**2)**0.25 * star['teff']  # K
        star['T_planet'] = star['T_planet'].decompose()

        if star['m_planet'] > 10 * q.M_earth:
            # Hydrogen (e.g. for gas giants)
            star['atm_molar_mass'] = 2.5 * q.g / q.mol
        else:
            # dry air (mostly nitrogen)  (e.g. for terrestial planets)
            star['atm_molar_mass'] = 29 * q.g / q.mol

        # assuming isothermal atmosphere, which is good enough on earth usually
        star['atm_scale_height'] = const.R * star['T_planet'] * star['r_planet']**2 / (
            const.G * star['m_planet'] * star['atm_molar_mass'])  # km
        star['atm_scale_height'] = star['atm_scale_height'].decompose()
        # effective atmosphere height, if it would have constant density
        star['h_atm'] = star['atm_scale_height'].to(q.km)

        logging.info('Planet Temperature: %s' % star['T_planet'])
        logging.info('Atmsophere Molar Mass: %s' %
                star['atm_molar_mass'])
        logging.info('Atmosphere Height: %s' % star['h_atm'])

        # calculate areas
        star['A_planet'] = np.pi * star['r_planet']**2
        star['A_star'] = np.pi * star['r_star']**2
        star['A_atm'] = np.pi * \
            (star['r_planet'] + star['h_atm'])**2 - star['A_planet']
        star['A_planet'] = star['A_planet'] / star['A_star']
        star['A_atm'] = star['A_atm'] / star['A_star']
        star['A_planet+atm'] = star['A_planet'] + star['A_atm']

        if 'periastron' not in star or star['periastron'] is None:
            star['periastron'] = star['transit']
        else:
            star["periastron"] = Time(star["periastron"], format="jd")

        return star
