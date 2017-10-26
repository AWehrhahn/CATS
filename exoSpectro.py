"""
Use an inverse method approach to determine the planetary transit spectrum
author: Ansgar Wehrhahn (ansgar.wehrhahn@physics.uu.se)
Based on work by Erik Aaronson (Uppsala University)
"""

import os.path
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt
from Config import config as Config

# Step 1: Load data
# Step 2: Calculate intermediary products F and G
# Step 3: Calculate planet spectrum

# Step 1: Load data
#   - Observation spectrum
#   - Stellar model
#   - Stellar and planetary orbit parameters
#   - Telluric spectrum


def load_observation():
    """ Load observation spectrum """
    obs_file = os.path.join(input_dir, config['file_observation'])
    obs = pd.read_table(obs_file, header=None, delim_whitespace=True)
    wl_grid = obs[0]
    del obs[0]
    return wl_grid, obs


def load_tellurics(wl_grid):
    """ Load telluric spectrum """
    tell_file = os.path.join(input_dir, config['file_telluric'])
    tell = pd.read_table(tell_file, header=None, delim_whitespace=True)
    # Skip interpolation, because they have the same wavelenght grid anyway
    print('Skip interpolation of tellurics as wavelength grid is the same')
    del tell[0]
    #tell = pd.DataFrame([interp1d(tell[0], tell[k], kind=config['interpolation_method'])(wl_grid) for k in range(1, tell.shape[1])])
    return tell


def load_parameters():
    """ Load orbital parameters """
    par_file = os.path.join(input_dir, config['file_parameters'])
    par = Config.load_yaml(par_file)

    # Convert all parameters into km and seconds
    r_sun = 696000
    r_jup = 71350
    au = 149597871
    secs = 24 * 60 * 60

    par['r_star'] = par['r_star'] * r_sun
    par['r_planet'] = par['r_planet'] * r_jup
    par['sma'] = par['sma'] * au
    par['period'] = par['period'] * secs
    par['duration'] = par['duration'] * secs
    return par


def load_star_model(wl_grid):
    """ Load stellar model data and apply normalization and wavelength interploation """
    # Prepare file names
    star_flux_file = os.path.join(input_dir, config['file_star_flux'])
    star_intensities = config['star_intensities']
    star_data_file = {i: os.path.join(
        input_dir, config['file_star_data'].format(str(i))) for i in star_intensities}

    # Skip first row as that wavelength is missing in intensity data
    star_flux = pd.read_table(star_flux_file, header=None,
                              delim_whitespace=True, usecols=(0, 1, 2), skiprows=(0, ))
    star_data = {i: pd.read_table(star_data_file[i], header=None, delim_whitespace=True, usecols=(
        1,)).values[:, 0] for i in star_intensities}
    # Assume that flux and intensity data are on the same wavelength grid
    #star_data['WL'] = pd.read_table(star_data_file[star_intensities[0]], delim_whitespace=True, header=None, usecols=(0,)).values[:,0]
    star_data = pd.DataFrame.from_dict(star_data)

    # Keep only interpolated spectrum
    # Discard data outside of wl_grid
    star_flux[0] = star_flux[0] * 0.1  # convert to Ångström
    # create normalization for star_data intensities
    # star_flux[1] is already normalized
    normalization = star_flux[2] / star_flux[1]
    star_data = star_data.apply(lambda x: x / normalization, axis=0)
    # Interpolate to wavelength grid

    def interpolation(d):
        """ interpolate d onto wl_grid """
        return interp1d(star_flux[0], d, kind=config['interpolation_method'])(wl_grid)

    tmp = {i: interpolation(star_data[i]) for i in star_intensities}
    star_data = pd.DataFrame.from_dict(tmp)
    star_flux = interpolation(star_flux[1])
    return star_flux, star_data


print('Loading data...')
# Load configuration file, most importantly includes paths to data directory
config = Config.load_config('config.yaml')

# exctract important parameters from config
target = config['name_target']
data_dir = os.path.join(config['path_exoSpectro'], target)

input_dir = os.path.join(data_dir, config['dir_input'])
output_dir = os.path.join(data_dir, config['dir_output'])

# Load Observation data
# Wavelength, obs1, obs2, .... ????
print('Loading observation data')
wl_grid, obs = load_observation()

# Load Telluric data
# Wavelength, obs1, obs2, .... ????
print('Loading telluric data')
tell = load_tellurics(wl_grid)

# Load stellar and planetary orbital parameters
print('Loading orbital parameters')
par = load_parameters()

# Load stellar model
# Wavelength, normalized Flux, Flux, ????
print('Loading stellar model')
star_flux, star_data = load_star_model(wl_grid)

print('Loading complete')

# Step 2: Calculate intermediate products F and G
# G = Telluric * I_atm
# F = -Obs + Flux_star * Telluric - (R_planet/R_star)**2 * Telluric * I_planet


def myvect(par):
    # TODO What does this describe actually?
    # Something about the position of the planet relative to the star

    i = par['inc'] * np.pi / 180
    d = np.sin(np.pi / 2 - i) * par['sma']
    total_distance = 2 * np.sqrt(par['r_star']**2 - d**2)

    distances = np.arange(par['n_exposures'], dtype=np.float) / (par['n_exposures'] - 1) * 2 * \
        (total_distance / 2 - par['r_planet']) / par['r_star']
    distances = distances - np.sum(distances) / len(distances)

    # distance from centre of stellar disk to centre of planet
    distance_center = np.sqrt(
        np.abs(d / par['r_star'])**2 + np.abs(distances)**2)

    #alternatively: np.sqrt(1-x**2)
    # but seriously why?
    return np.cos(np.arcsin(distance_center))


def atmosphere_calc(par, my, n):
    # TODO figure out what is happening here
    d = np.sin(np.arccos(my)) #distance from the center of the stellar disk to the centre of the planet
    r = (par['r_planet'] + par['h_atm'] / 2) / par['r_star']

    phi = np.arange(n, dtype=np.float) / n * 2 * np.pi
    x = np.sqrt(d**2 + r**2 - 2 * d * r * np.cos(phi)) #Some kind of vector thingy

    # Use the where keyword to avoid unnecessary calculations
    tmp = x < 1
    return np.where(tmp, np.cos(np.arcsin(x, where=tmp), where=tmp), 0)


def intensity_interpolation(my, i):
    """ Interpolate values of my between starI files i """
    return interp1d(i.keys(), i.values, kind=config['interpolation_method'], fill_value=0, bounds_error=False)(my)


def intensity_atmosphere(par, my, star_data, n=20):
    """
    Calculate the specific intensties blocked by the planetary atmosphere
    par: paramters
    my: my value
    star_data: star intensity data
    n: number of points in the atmosphere
    """
    # if my is a vector return a list of values
    if isinstance(my, (list, np.ndarray)):
        return np.array([intensity_atmosphere(par, my[k], star_data, n=n) for k in range(len(my))])

    atmo = atmosphere_calc(par, my, n)
    # create mean intensity spectra for all atmosphere points
    i = intensity_interpolation(atmo, star_data)
    i = np.sum(i, axis=1)
    return i / n


print('Calculate intermidiary products')
my = myvect(par)
print('Calculate specific intensities for the planetary atmosphere')
I_atm = intensity_atmosphere(par, my, star_data, n=20)
#TODO shift to exoplanet restframe

plt.plot(wl_grid, I_atm[0, :])
plt.show()

pass
# I_planet =
