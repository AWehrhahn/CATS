"""
Get Data from Stellar DB
"""

import numpy as np
from data_module_interface import data_module
from DataSources.StellarDB import StellarDB

class stellar_db(data_module):
    @classmethod
    def load_parameters(cls, name_star, name_planet, atm_factor=0.1, **kwargs):
        """ loads the stellar parameters from the StellarDB database

        Converts measurements into km, s, and radians and calculates some simple fixed parameters like the projected area of the planet on the star
        See StellarDB project for more details

        Parameters:
        ----------
        name_star : {str}
            name of the star, if not found in local StellarDB it will automatically be resolved with SIMBAD
        name_planet : {str}
            name of the planet, just the letter, e.g. 'b'
        **kwargs

        atm_factor : {float}, optional
            size of the atmosphere in relation ro planet radius (the default is 0.1)

        """
        cls.log(2, 'Stellar DB')
        sdb = StellarDB()
        star = sdb.load(name_star)
        # Convert names
        # Stellar parameters
        star['name_star'] = star['name'][0]
        star['r_star'] = star['radius']
        star['m_star'] = star['mass']
        star['star_temp'] = star['t_eff']
        star['star_logg'] = star['logg']
        star['star_vt'] = star['vel_turb']
        star['star_metallicity'] = star['metallicity']
        # Planetary parameters
        planet = star['planets'][name_planet]
        star['name_planet'] = name_planet
        star['r_planet'] = planet['radius']
        star['inc'] = planet['inclination']
        if 'atm_height' in planet.keys():
            star['h_atm'] = planet['atm_height']
        star['sma'] = planet['semi_major_axis']
        star['period'] = planet['period']
        star['transit'] = planet['transit_epoch']
        star['duration'] = planet['transit_duration']
        star['eccentricity'] = planet['eccentricity']

        # Convert all parameters into km and seconds
        r_sun = 696342      # Radius Sun
        r_jup = 69911       # Radius Jupiter
        au = 149597871      # Astronomical Unit
        secs = 24 * 60 * 60  # Seconds in a day
        m_sol = 1.98855e30 #kg

        star['m_star'] *= m_sol
        star['r_star'] = star['r_star'] * r_sun
        star['r_planet'] = star['r_planet'] * r_jup
        star['sma'] = star['sma'] * au
        star['period_s'] = star['period'] * secs
        star['period_h'] = star['period'] * 24
        star['duration'] = star['duration'] * secs

        # Convert to radians
        star['inc'] = np.deg2rad(star['inc'])

        # Derived values, the pi factor gets canceled out
        # TODO get a better estimate/value
        # if 'h_atm' not in star.keys():
        #    star['h_atm'] = 0.1 * star['r_planet']
        # else:
        star['h_atm'] = atm_factor * star['r_planet']

        star['A_planet'] = star['r_planet']**2
        star['A_star'] = star['r_star']**2
        star['A_atm'] = (star['r_planet'] + star['h_atm'])**2 - star['A_planet']
        star['A_planet'] = star['A_planet'] / star['A_star']
        star['A_atm'] = star['A_atm'] / star['A_star']
        star['A_planet+atm'] = star['A_planet'] + star['A_atm']

        if 'periastron' not in star or star['periastron'] is None:
            star['periastron'] = star['transit']

        return star
