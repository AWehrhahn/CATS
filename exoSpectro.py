"""
Use an inverse method approach to determine the planetary transit spectrum
author: Ansgar Wehrhahn (ansgar.wehrhahn@physics.uu.se)
Based on work by Erik Aaronson (Uppsala University)
"""

import os.path
import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import solve_banded
import pandas as pd
import matplotlib.pyplot as plt
from Config import config as Config

# Step 1: Load data
# Step 2: Calculate intermediary products F and G
# Step 3: Calculate planet spectrum


# Load configuration file, most importantly includes paths to data directory
config = Config.load_config('config.yaml')

# exctract important parameters from config
target = config['name_target']
data_dir = os.path.join(config['path_exoSpectro'], target)

input_dir = os.path.join(data_dir, config['dir_input'])
output_dir = os.path.join(data_dir, config['dir_output'])

# skip calculation of F and G if they already exist
intermediary_file = os.path.join(output_dir, config['file_intermediary'])
if os.path.exists(intermediary_file):
    print('Load existing intermediary data F and G')
    intermediary_file = np.load(intermediary_file)
    F = intermediary_file['F']
    G = intermediary_file['G']
    wl_grid = intermediary_file['wl']

else:
    # Step 1: Load data
    #   - Stellar and planetary orbit parameters
    #   - Observation spectrum
    #   - Telluric spectrum
    #   - Stellar model

    def load_observation():
        """ Load observation spectrum """
        obs_file = os.path.join(input_dir, config['file_observation'])
        obs = pd.read_table(obs_file, header=None, delim_whitespace=True)
        wl_grid = obs[0]
        obs.drop([0, *range(par['n_exposures'] + 1, obs.shape[1])],
                 axis=1, inplace=True)
        obs = obs.swapaxes(0, 1)
        return wl_grid, obs

    def load_tellurics(wl_grid):
        """ Load telluric spectrum """
        tell_file = os.path.join(input_dir, config['file_telluric'])
        tell = pd.read_table(tell_file, header=None, delim_whitespace=True)
        tell.drop([0, *range(par['n_exposures'] + 1, tell.shape[1])],
                  axis=1, inplace=True)
        # Skip interpolation, because they have the same wavelenght grid anyway
        print('Skip interpolation of tellurics as wavelength grid is the same')
        #tell = pd.DataFrame([interp1d(tell[0], tell[k], kind=config['interpolation_method'])(wl_grid) for k in range(1, tell.shape[1])])
        tell = tell.swapaxes(0, 1)
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

    # Load stellar and planetary orbital parameters
    print('Loading orbital parameters')
    par = load_parameters()

    # Load Observation data
    # Wavelength, obs1, obs2, .... ????
    print('Loading observation data')
    wl_grid, obs = load_observation()

    # Load Telluric data
    # Wavelength, obs1, obs2, .... ????
    print('Loading telluric data')
    tell = load_tellurics(wl_grid)

    # Load stellar model
    # Wavelength, normalized Flux, Flux, ????
    print('Loading stellar model')
    star_flux, star_data = load_star_model(wl_grid)

    print('Loading complete')

    # Step 2: Calculate intermediate products F and G
    # G = Telluric * I_atm
    # F = -Obs + Flux_star * Telluric - (R_planet/R_star)**2 * Telluric * I_planet

    def distance_transit(par):
        """
        Calculate the distances from the centre of the stellar disk 
        to the centre of the planet along the transit
        par: Orbital parameters
        """

        """
            |\ 
        d | \ 
            |  \ r_s
            |__i\ 
            td
        """

        i = np.deg2rad(par['inc'])
        # distance from the ecliptic; i = 90 degree : Edge on orbit, transit along the ecliptic
        d = np.cos(i) * par['sma']
        # half of the total distance traveled over the star
        total_distance = np.sqrt(par['r_star']**2 - d**2)

        # split the total traveled distance into n equally spaced parts, where n is the number of exposures
        distances = np.arange(par['n_exposures'], dtype=np.float) / (par['n_exposures'] - 1) * 2 * \
            (total_distance - par['r_planet']) / par['r_star']
        # relative to the center of transit
        distances = distances - np.sum(distances) / len(distances)

        # distance from centre of stellar disk to centre of planet
        distance_center = np.sqrt(
            np.abs(d / par['r_star'])**2 + np.abs(distances)**2)

        return distance_center

    def calc_distances(par, sample_radius, d, n):
        """
        Calculate distances of points in the atmosphere/planet to the center of the stallar disk
        par: Orbital parameters
        sample_radius: radius from the planet center to sample 
        d: distances from stellar center along the orbit
        n: number of points in the atmosphere
        """
        if isinstance(sample_radius, np.ndarray):
            return np.array([calc_distances(par, s, d, n) for s in sample_radius])
        # size of the planet (with atmosphere) in stellar radii
        # sample radius = (par['r_planet'] + par['h_atm'] / 2)
        r = sample_radius / par['r_star']

        # whole circle split into n parts
        phi = np.arange(n, dtype=np.float) / n * 2 * np.pi
        # law of cosines, distances to different points in the atmosphere
        x = np.sqrt(d**2 + r**2 - 2 * d * r * np.cos(phi))

        # if x > r_star then return r_star
        # Use the where keyword to avoid unnecessary calculations
        tmp = x < 1  # smaller than r_star
        return np.where(tmp, x, 1)

    def intensity_interpolation(dt, i):
        """ Interpolate values of distances between starI files i """
        # Switch axes ????
        d = np.sqrt(1 - dt**2)  # cos(arcsin(dt))
        return interp1d(i.keys(), i.values, kind=config['interpolation_method'], fill_value=0, bounds_error=False)(d)

    def calc_specific_intensity(par, dt, radii, star_data, n=20):
        """
        Calculate specific intensities blocked by solid planet body
        par: Orbital parameters
        dt: planet-star distances during transit
        radii: distances from the center of the planet (in km) to sample
        star_data: stellar intensity data
        n: number of angles to sample
        """
        if isinstance(dt, np.ndarray):
            return np.array([calc_specific_intensity(par, k, radii, star_data, n=n) for k in dt])

        distance_planet = calc_distances(par, radii, dt, n)
        # outer radii should have larger weight in the average due to larger area contribution
        # weights scale linearly with radius because dA = 2*pi*r*dr, if we assume dr << r

        i = np.array([intensity_interpolation(
            distance_planet[j, :], star_data) for j in range(len(radii))])
        i = np.mean(i, axis=2)
        i = np.average(i, axis=0, weights=radii)
        return i

    def intensity_atmosphere(par, dt, star_data, n=20):
        """
        Calculate the specific intensties blocked by the planetary atmosphere
        par: paramters
        dt: planet-star distances during transit
        star_data: star intensity data
        n: number of points in the atmosphere
        """
        # Sample the center of the atmosphere
        sample_radius = np.array([par['r_planet'] + par['h_atm'] / 2])
        return calc_specific_intensity(par, dt, sample_radius, star_data, n=n)

    def intensity_planet(par, dt, star_data, n=20, m=30):
        """
        Calculate specific intensities blocked by solid planet body
        par: Orbital parameters
        dt: planet-star distances during transit
        star_data: stellar intensity data
        n: number of angles to sample
        m: number of radii to sample
        """
        # various distances from the centre of the planet sampled here
        radii = np.arange(m, dtype=np.float) / m * par['r_planet']
        return calc_specific_intensity(par, dt, radii, star_data, n=n)

    def doppler_shift(spectrum, wl, vel):
        """ Shift spectrum by velocity vel """
        if isinstance(vel, np.ndarray) and isinstance(spectrum, np.ndarray):
            if spectrum.ndim > 1:
                return np.array([doppler_shift(spectrum[k], wl, vel[k]) for k in range(len(vel))])
            else:
                return np.array([doppler_shift(spectrum, wl, vel[k]) for k in range(len(vel))])
        c0 = 299792  # speed of light in km/s
        # new shifted wavelength grid
        wl_doppler = wl * (1 + vel / c0)
        return interp1d(wl, spectrum, kind=config['interpolation_method'], fill_value='extrapolate')(wl_doppler)

    def rv_star(par):
        """ linearly distribute radial velocities during transit """
        return np.arange(par['n_exposures']) / (par['n_exposures'] - 1.) * \
            (par['rv_end'] - par['rv_start']) + par['rv_start']

    def rv_planet(par):
        """ calculate radial velocities of the planet along the orbit """
        i = np.deg2rad(par['inc'])
        v_orbit = par['sma'] * 2 * np.pi / par['period']
        # each exposures time from periastron
        sec_exposure = np.arange(par['n_exposures']) / \
            (par['n_exposures'] - 1) - 0.5 * par['duration']

        # angle of full orbit from periastron that each exposure is taken
        angle_exposure = sec_exposure / par['period'] * 2 * np.pi
        # Calculate radial velocities
        vel_p = np.abs(v_orbit * np.arctan(angle_exposure) * np.sin(i))

        # invert velocities of second half
        vel_p[par['n_exposures'] // 2:] = - vel_p[par['n_exposures'] // 2:]

        return vel_p

    print('Calculate intermidiary products')
    dt = distance_transit(par)
    # radial velocities during transit, linearly distributed
    # relative to star ??
    # TODO check if doppler shifts are correct
    vel_b = rv_star(par)
    vel_p = rv_planet(par)

    print('Calculate specific intensities blocked by planetary atmosphere')
    I_atm = intensity_atmosphere(par, dt, star_data, n=20)
    I_atm = doppler_shift(I_atm, wl_grid, vel_b + vel_p)

    print('Calculate specific intensities blocked by solid planet body')
    # TODO I changed the radii at which the intensity is calculated check if that is correct or not
    I_planet = intensity_planet(par, dt, star_data, n=20, m=30)
    I_planet = doppler_shift(I_planet, wl_grid, vel_b + vel_p)

    print('Doppler shift stellar flux')
    star_flux = doppler_shift(star_flux, wl_grid, vel_b + vel_p)

    print('Calculate G = Telluric * I_atmosphere')
    G = tell * I_atm
    # leave pandas dataframes behind and only work with numpy arrays afterwards
    G = G.values

    print('Calculate F = -Observation + Flux * Telluric - (R_planet/R_star)**2 * I_planet * Telluric')
    F = - obs + star_flux * tell - \
        (par['r_planet'] / par['r_star'])**2 * I_planet * tell
    F = F.values

    # TODO Brightness correction parameter
    # (r(v) ccr star_flux) * (r(v) ccr tellurics) = min, where ccr is the cross-correlation operator

    # save F and G for further use
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.savez(intermediary_file, F=F, G=G, wl=wl_grid)

# Step 3: Calculate planet spectrum
#   - try different values for regularization parameter lambda


def solve(wl_grid, F, G, lam):
    """
    Solve the mimimazation problem to find the planetary spectrum
    wl_grid: Wavelength scale
    F: intermediary product F
    G: intermediary product G
    lam: regularization parameter lambda
    """
    delta_wl = 1 / (wl_grid[1] - wl_grid[0])**2

    # Solve the tridiagonal problem [a,b,c] * x = r
    a = np.full(G.shape[1], - lam * delta_wl)
    b = np.sum(G**2, axis=0) + 2 * delta_wl * lam
    r = np.sum(F * G, axis=0)

    # First and last element only have one other element in their respective sums
    # Therefore compensate by removing something like it
    b[0] -= lam * delta_wl
    b[-1] -= lam * delta_wl

    ab = np.array([a, b, a])
    return solve_banded((1, 1), ab, r)


def normalize(a, axis=-1, order=2):
    """ normalize array a along axis """
    norm = np.max(np.abs(a))
    return a / norm


print('Solve minimization problem for planetary spectrum')
# TODO try to find best value for lambda
# Brute Force Solution: try different values for lambda and find the best
# What is the best lambda ??? Which metric is used to determine that?
solution = solve(wl_grid, F, G, 1000)
# TODO normalize
solution = normalize(solution)

# Step 4: Profit, aka Plotting
plt.plot(wl_grid, solution)
plt.show()
