"""
Get Data from Stellar DB
"""

import numpy as np
from scipy import constants as const
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
        star['m_planet'] = planet['mass']
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
        au = 149597871      # Astronomical Unit287.058
        secs = 24 * 60 * 60  # Seconds in a day
        m_sol = 1.98855e30  # kg
        m_jup = 1.89813e27  # kg
        m_earth = 5.972e24  # kg

        star['m_star'] *= m_sol
        star['m_planet'] *= m_jup
        star['r_star'] = star['r_star'] * r_sun
        star['r_planet'] = star['r_planet'] * r_jup
        star['sma'] = star['sma'] * au
        star['period_s'] = star['period'] * secs
        star['period_h'] = star['period'] * 24
        star['duration'] = star['duration'] * secs

        # Convert to radians
        star['inc'] = np.deg2rad(star['inc'])

        # TODO: atmosphere model
        # stellar flux in = thermal flux out
        star['T_planet'] = ((np.pi * star['r_planet']**2) /
                            star['sma']**2)**0.25 * star['t_eff']  # K

        if star['m_planet'] > 10 * m_earth:
            # Hydrogen (e.g. for gas giants)
            star['atm_molar_mass'] = 2
        else:
            # dry air (mostly nitrogen)  (e.g. for terrestial planets)
            star['atm_molar_mass'] = 29

        # assuming isothermal atmosphere, which is good enough on earth usually
        star['atm_scale_height'] = const.gas_constant * star['T_planet'] * (star['r_planet'] * 1e3)**2 / (
            const.gravitational_constant * star['m_planet'] * star['atm_molar_mass'])  # km
        # effective atmosphere height, if it would have constant density
        star['h_atm'] = star['atm_scale_height']

        cls.log(2, 'Planet Temperature: %.2f K' % star['T_planet'])
        cls.log(2, 'Atmsophere Molar Mass: %.2f g/mol' %
                star['atm_molar_mass'])
        cls.log(2, 'Atmosphere Height: %.2f km' % star['h_atm'])

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

        return star
