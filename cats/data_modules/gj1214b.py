import numpy as np
from scipy import constants as const
from .data_interface import data_orbitparameters

rsun = 695510 # km
msun = 1.989e30 # kg
au = 1.496e8 # km
mjup = 1.89813e27 # kg
m_earth = 5.972e24  # kg


class gj1214b(data_orbitparameters):
    def get_parameters(self, **data):
        parameters = {
            "period": 1.58040482, #days
            "periastron": 2454980.748796, # jd
            "transit": 2454980.748796, # jd
            "t_eff": 3030, # K
            "r_star" : 	0.211 * rsun, # r_sun
            "m_star": 0.157 * msun, # m_sun
            "m_planet": 0.0204 * mjup, # m_jup
            "sma": 0.01433 * au, # au
            "inc": 90, # degree
            "eccentricity": 0, 
            "w" : 90
        }

        parameters["r_planet"] =  np.sqrt(0.0135) * parameters["r_star"]
        
        if parameters['m_planet'] > 10 * m_earth:
            # Hydrogen (e.g. for gas giants)
            parameters['atm_molar_mass'] = 2.5
        else:
            # dry air (mostly nitrogen)  (e.g. for terrestial planets)
            parameters['atm_molar_mass'] = 29

        parameters["T_planet"] = ((np.pi * parameters['r_planet']**2) / parameters['sma']**2)**0.25 * parameters['t_eff']  # K
        parameters["h_atm"] = const.gas_constant * parameters['T_planet'] * (parameters['r_planet'] * 1e3)**2 / (
            const.gravitational_constant * parameters['m_planet'] * parameters['atm_molar_mass'])  # km

        parameters['A_planet'] = np.pi * parameters['r_planet']**2
        parameters['A_star'] = np.pi * parameters['r_star']**2
        parameters['A_atm'] = np.pi * \
            (parameters['r_planet'] + parameters['h_atm'])**2 - parameters['A_planet']
        parameters['A_planet'] = parameters['A_planet'] / parameters['A_star']
        parameters['A_atm'] = parameters['A_atm'] / parameters['A_star']
        parameters['A_planet+atm'] = parameters['A_planet'] + parameters['A_atm']

        return parameters
