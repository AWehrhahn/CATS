"""
Use an inverse method approach to determine the planetary transit spectrum
author: Ansgar Wehrhahn (ansgar.wehrhahn@physics.uu.se)
Based on work by Erik Aaronson (Uppsala University)
"""

import os.path
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
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
    return obs


def load_tellurics():
    """ Load telluric spectrum """
    tell_file = os.path.join(input_dir, config['file_telluric'])
    tell = pd.read_table(tell_file, header=None, delim_whitespace=True)
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
        input_dir, config['file_star_data'].format(i)) for i in star_intensities}

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
        return interp1d(star_flux[0], d)(wl_grid)

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
obs = load_observation()
wl_grid = obs[0]

# Load Telluric data
# Wavelength, obs1, obs2, .... ????
print('Loading telluric data')
tell = load_tellurics()

# Load stellar and planetary orbital parameters
print('Loading orbital parameters')
par = load_parameters()

# Load stellar model
# Wavelength, normalized Flux, Flux, ????
print('Loading stellar model')
star_flux, star_data = load_star_model(wl_grid)

print('Loading complete')

# Step 2: Calculate intermidiate products F and G
# G = Telluric * I_atm
# F = -Obs + Flux_star * Telluric - (R_planet/R_star)**2 * Telluric * I_planet


def myvect(par):
    # TODO What does this describe actually?
    # Something about the position of the planet relative to the star
    n = par['n_exposures']
    inc = par['inc']
    sma = par['sma']
    rs = par['r_star']
    rp = par['r_planet']

    i = inc * np.pi / 180
    d = np.sin(np.pi / 2 - i) * sma
    total_distance = 2 * np.sqrt(rs**2 - d**2)

    distances = np.arange(n, dtype=np.float) / (n - 1) * 2 * \
        (total_distance / 2 - rp) / rs
    distances = distances - np.sum(distances) / len(distances)

    # distance from centre of stellar disk to centre of planet
    distance_center = np.sqrt(
        np.abs(d / rs)**2 + np.abs(distances)**2)

    #alternatively: np.sqrt(1-x**2)
    # but seriously why?
    return np.cos(np.arcsin(distance_center))


def atmosphere_calc(par, my, n):
    d = np.sin(np.arccos(my))
    r = (par['r_planet'] + par['h_atm'] / 2) / par['r_star']

    phi = np.arange(n, dtype=np.float) / n * 2 * np.pi
    x = np.sqrt(d**2 + r**2 - 2 * d * r * np.cos(phi))

    return np.where(x < 1, np.cos(np.arcsin(x)), 0)


def intensity_interpolation(my, i):
    # Interpolate between two starI files ???

    # find nearest files
    myi = 10 * my
    i1 = np.floor(myi)
    i2 = np.ceil(myi)
    diff = myi - i1

    factor_1 = 1 - diff
    factor_2 = diff

    return i[i1, :] * factor_1 + i[i2, :] * factor_2


def intensity_atmosphere(par, my, wl, star_data, n=20):
    # par: paramters
    # my: my value (not vector)
    # wl: wavelenght grid
    # star_data: star intensity data
    # n: number of points in the atmosphere
    atmo = atmosphere_calc(par, my, n)
    # create mean intensity spectra for all atmosphere points
    i = np.zeros(len(wl), dtype=np.float)
    for k in range(int(n)):
        i = i + intensity_interpolation(atmo[k], star_data)
    return i / n


print('Calculate intermidiary products')
my = myvect(par)
for n in range(par['n_exposures']):
    I_atm = intensity_atmosphere(par, my[n], wl_grid, star_data, n=20)
# I_planet =
