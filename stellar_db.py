"""
Get Data from Stellar DB
"""

import numpy as np
from DataSources.StellarDB import StellarDB

def load_parameters(name_star, name_planet, atm_factor=0.1, **kwargs):
    """ Load orbital parameters """

    sdb = StellarDB()
    star = sdb.load(name_star)
    # Convert names
    # Stellar parameters
    star['name_star'] = star['name'][0]
    star['r_star'] = star['radius']
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

    # Convert all parameters into km and seconds
    r_sun = 696000      # Radius Sun
    r_jup = 71350       # Radius Jupiter
    au = 149597871      # Astronomical Unit
    secs = 24 * 60 * 60  # Seconds in a day

    star['r_star'] = star['r_star'] * r_sun
    star['r_planet'] = star['r_planet'] * r_jup
    star['sma'] = star['sma'] * au
    star['period_s'] = star['period'] * secs
    star['duration'] = star['duration'] * secs

    # Convert to radians
    star['inc'] = np.deg2rad(star['inc'])

    # Derived values, the pi factor gets canceled out
    # TODO get a better estimate/value
    if 'h_atm' not in star.keys():
        star['h_atm'] = 0.1 * star['r_planet']
    else:
        star['h_atm'] = atm_factor * star['r_planet']

    star['A_planet'] = star['r_planet']**2
    star['A_star'] = star['r_star']**2
    star['A_atm'] = (star['r_planet'] + star['h_atm'])**2 - star['A_planet']
    star['A_planet'] = star['A_planet'] / star['A_star']
    star['A_atm'] = star['A_atm'] / star['A_star']
    star['A_planet+atm'] = star['A_planet'] + star['A_atm']

    return star