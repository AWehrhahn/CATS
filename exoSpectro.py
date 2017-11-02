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


# Load configuration file, most importantly includes paths to data directory
config = Config.load_config('config.yaml')

# exctract important parameters from config
target = config['name_target']
data_dir = os.path.join(config['path_exoSpectro'], target)

input_dir = os.path.join(data_dir, config['dir_input'])
output_dir = os.path.join(data_dir, config['dir_output'])

# skip calculation of F and G if they already exist
intermediary_file = os.path.join(output_dir, config['file_intermediary'])
renew_all = config['renew_all']

print('Loading data...')

# Load stellar and planetary orbital parameters
print('Loading orbital parameters')
par = load_parameters()

if os.path.exists(intermediary_file) and not renew_all:
    print('Load existing intermediary data F and G')
    intermediary_file = np.load(intermediary_file)
    F = intermediary_file['F']
    G = intermediary_file['G']
    wl_grid = intermediary_file['wl']
    tell = intermediary_file['tell']
    obs = intermediary_file['obs']

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
        wl_tmp = obs[0].values
        obs.drop([0, *range(par['n_exposures'] + 1, obs.shape[1])],
                 axis=1, inplace=True)
        obs = obs.values.swapaxes(0, 1)
        #obs = interp1d(wl_tmp, obs, kind=config['interpolation_method'], fill_value='extrapolate')(wl_grid)
        return obs, wl_tmp

    def load_tellurics(wl_grid):
        """ Load telluric spectrum """
        tell_file = os.path.join(input_dir, config['file_telluric'])
        tell = pd.read_table(tell_file, header=None, delim_whitespace=True)
        wl_tmp = tell[0]
        tell.drop([0, *range(par['n_exposures'] + 1, tell.shape[1])],
                  axis=1, inplace=True)
        # Skip interpolation, because they have the same wavelenght grid anyway
        #print('Skip interpolation of tellurics as wavelength grid is the same')
        tell = tell.values.swapaxes(0, 1)
        tell = interp1d(
            wl_tmp, tell, kind=config['interpolation_method'], fill_value='extrapolate')(wl_grid)
        return tell

    def load_star_model(wl_grid):
        """ Load stellar model data and apply normalization and wavelength interploation """
        # Prepare file names
        star_flux_file = os.path.join(input_dir, config['file_star_flux'])
        star_intensities = config['star_intensities']
        star_data_file = {i: os.path.join(
            input_dir, config['file_star_data'].format(str(i))) for i in star_intensities}

        star_flux = pd.read_table(star_flux_file, header=None,
                                  delim_whitespace=True, usecols=(0, 1, 2)).values
        star_data = {i: pd.read_table(star_data_file[i], header=None, delim_whitespace=True, usecols=(0,
                                                                                                      1,)).values for i in star_intensities}

        # fix wavelenghts
        star_flux[:, 0] = star_flux[:, 0] * 0.1  # convert to Ångström

        for i in star_intensities:
            star_data[i][:, 0] *= 0.1

        # Interpolate to wavelength grid
        def interpolation(wl_old, spec, wl_new):
            """ interpolate d onto wl_grid """
            return interp1d(wl_old, spec, kind=config['interpolation_method'], fill_value='extrapolate')(wl_new)

        def instrument_profile(par, spectrum):
            """ apply instrumental profile broadening to the spectrum """
            height = 0.08
            # x = -width ... +width
            x = np.arange(-par['width'], par['width'] +
                          1, step=1, dtype=np.float)
            y = height * np.exp(-0.5 * (x * 2.67 / par['fwhm'])**2)  # Gaussian

            extspec = np.zeros(len(spectrum) + 2 * par['width'], dtype=float)
            extspec[:par['width']] = spectrum[0]
            extspec[par['width']:-par['width']] = spectrum
            extspec[-par['width']:] = spectrum[-1]

            outspec = np.zeros(len(spectrum))
            for i in range(len(spectrum)):
                outspec[i] = np.sum(
                    extspec[i:i + 2 * par['width'] + 1] * y)

            normali = np.sum(spectrum[par['width']:-par['width']]) / \
                np.sum(outspec[par['width']:-par['width']])
            outspec = outspec * normali
            return outspec

        normalization = interpolation(
            star_flux[:, 0], star_flux[:, 1] / star_flux[:, 2], wl_grid)
        star_flux = interpolation(
            star_flux[:, 0], star_flux[:, 2], wl_grid) * normalization
        star_flux = instrument_profile(par, star_flux)

        tmp = {i: interpolation(
            star_data[i][:, 0], star_data[i][:, 1], wl_grid) for i in star_intensities}
        for i in star_intensities:
            tmp[i] *= normalization
            tmp[i] = instrument_profile(par, tmp[i])

        tmp[0.0] = np.zeros(len(wl_grid), dtype=float)
        star_data = pd.DataFrame.from_dict(tmp)
        #star_flux = pd.DataFrame(star_flux)

        return star_flux, star_data

    # Load Observation data
    # Wavelength, obs1, obs2, ....
    print('Loading observation data')
    obs, wl_grid = load_observation()

    # Load Telluric data
    # Wavelength, obs1, obs2, ....
    print('Loading telluric data')
    tell = load_tellurics(wl_grid)

    # Load stellar model
    # Wavelength, normalized Flux, Flux, ...
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

        r"""
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
        distances = np.linspace(
            0, 2 * (total_distance - par['r_planet']) / par['r_star'], par['n_exposures'])

        # relative to the center of transit
        distances = distances - np.mean(distances)

        # distance from centre of stellar disk to centre of planet
        distance_center = np.sqrt((d / par['r_star'])**2 + (distances)**2)

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
        phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
        # law of cosines, distances to different points in the atmosphere
        x = np.sqrt(d**2 + r**2 - 2 * d * r * np.cos(phi))

        # if x > r_star then return r_star
        # Use the where keyword to avoid unnecessary calculations
        x[x > 1] = 1
        return x

    def intensity_interpolation(dt, i):
        """ Interpolate values of distances between starI files i """
        d = np.sqrt(1 - dt**2)  # cos(arcsin(dt))
        return interp1d(i.keys().values, i.values, kind=config['interpolation_method'], fill_value='extrapolate', bounds_error=False)(d)

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
        weight = (2 * np.arange(len(radii)) + 1) / len(radii)**2
        i = np.sum(i * weight[:, None], axis=0)
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

    def intensity_planet(par, dt, star_data, n=20, m=20):
        """
        Calculate specific intensities blocked by solid planet body
        par: Orbital parameters
        dt: planet-star distances during transit
        star_data: stellar intensity data
        n: number of angles to sample
        m: number of radii to sample
        """
        # various distances from the centre of the planet sampled here
        radii = np.linspace(1 + 0.25 / m, 1.25, m) * \
            (par['r_planet'] + par['h_atm'])
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
        return interp1d(wl_doppler, spectrum, kind=config['interpolation_method'], fill_value=0, bounds_error=False)(wl)

    def rv_star(par):
        """ linearly distribute radial velocities during transit """
        return np.linspace(0, par['rv_end'] - par['rv_start'], par['n_exposures']) + par['rv_start']

    def rv_planet(par):
        """ calculate radial velocities of the planet along the orbit """
        # TODO use times from individual observations, i.e. the values in the fits headers, instead of assuming equidistant spread
        i = np.deg2rad(par['inc'])
        v_orbit = par['sma'] * 2 * np.pi / par['period']

        # angle of full orbit from periastron that each exposure is taken, exposure time / period
        angle_exposure = np.linspace(-np.pi, np.pi,
                                     par['n_exposures']) * par['duration'] / par['period']
        # Calculate radial velocities
        vel_p = np.abs(v_orbit * np.arctan(angle_exposure) * np.sin(i))

        # invert velocities of second half
        vel_p[par['n_exposures'] // 2:] = - vel_p[par['n_exposures'] // 2:]

        return vel_p

    print('Calculate intermediary products')
    dt = distance_transit(par)
    # radial velocities during transit, linearly distributed
    # relative to star ??
    # TODO check if doppler shifts are correct
    vel_b = rv_star(par)
    vel_p = rv_planet(par)

    print('Calculate specific intensities blocked by planetary atmosphere')
    # TODO find more efficient way to calculate the specific intensities
    I_atm = intensity_atmosphere(par, dt, star_data, n=20)
    I_atm = doppler_shift(I_atm, wl_grid, vel_b)

    print('Calculate specific intensities blocked by solid planet body')
    # TODO I changed the radii at which the intensity is calculated check if that is correct or not
    I_planet = intensity_planet(par, dt, star_data, n=20, m=20)
    I_planet = doppler_shift(I_planet, wl_grid, vel_b)

    print('Doppler shift stellar flux')
    star_flux = doppler_shift(star_flux, wl_grid, vel_b)

    print('Calculate G = Telluric * I_atmosphere')
    # TODO I_atm has lower maximum value than it should have, but correct shape
    G = tell * I_atm
    G = doppler_shift(G, wl_grid, -vel_b - vel_p)

    print('Calculate F = -Observation + Flux * Telluric - (R_planet/R_star)**2 * I_planet * Telluric')
    F = - obs + star_flux * tell - \
        (par['r_planet'] / par['r_star'])**2 * I_planet * tell
    F = doppler_shift(F, wl_grid, -vel_b - vel_p)

    # TODO Brightness correction parameter
    # (r(v) ccr star_flux) * (r(v) ccr tellurics) = min, where ccr is the cross-correlation operator

    # save F and G for further use
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.savez(intermediary_file, F=F, G=G, wl=wl_grid, tell=tell, obs=obs)

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
    delta_wl = np.zeros_like(wl_grid)
    delta_wl[1:] = 1 / (wl_grid[1:] - wl_grid[:-1])**2
    delta_wl[0] = delta_wl[1]

    # Solve the tridiagonal problem [a,b,c] * x = r
    a = - lam * delta_wl
    b = np.mean(G**2, axis=0) * par['n_exposures'] + 2 * delta_wl * lam
    b = b.reshape(b.shape[-1])
    r = np.mean(F * G, axis=0) * par['n_exposures']
    r = r.reshape(r.shape[-1])

    # First and last element only have one other element in their respective sums
    # Therefore compensate by removing something like it
    b[0] -= lam * delta_wl[0]
    b[-1] -= lam * delta_wl[-1]

    ab = np.array([a, b, a])
    return solve_banded((1, 1), ab, r)


def normalize(a):
    """ normalize array a along axis """
    a = np.abs(a)
    a -= np.min(a)
    norm = np.max(a)
    return a / norm


print('Alternative questionable apporach')
exo2 = pd.read_table(os.path.join(
    input_dir, par['file_atmosphere']), header=None, delim_whitespace=True).values
exo2 = interp1d(exo2[:, 0], exo2[:, 1], kind=config['interpolation_method'],
                fill_value='extrapolate')(wl_grid)
exo2 = exo2 * par['h_atm'] - par['h_atm']

Fexo = np.array([[exo2 * G[1, :]], [exo2 * G[5, :]], [exo2 * G[8, :]]])
gtemp = np.full(exo2.shape, 1)
Gexo = np.array([[gtemp * G[1, :]], [gtemp * G[5, :]], [gtemp * G[8, :]]])
nexp = F.shape[0]
lambdaexo = 1500 * 3 / nexp
exo = solve(wl_grid, Fexo, Gexo, lambdaexo)
# Normalize
exo = (exo - np.min(exo)) / np.max(exo - np.min(exo))
exo2 = (exo2 - np.min(exo2)) / np.max(exo2 - np.min(exo2))

plt.plot(wl_grid, exo2)
plt.plot(wl_grid, exo, 'r')
plt.show()

print('Solve minimization problem for planetary spectrum')
# TODO try to find best value for lambda
# Brute Force Solution: try different values for lambda and find the best
# What is the best lambda ??? Which metric is used to determine that?
solution = solve(wl_grid, F, G, 1500)
solution = normalize(solution)

# Step 4: Profit, aka Plotting
input_file = os.path.join(input_dir, par['file_atmosphere'])
input_spectrum = pd.read_table(input_file, header=None, delim_whitespace=True).values.swapaxes(0,1)
input_spectrum[1] = normalize(input_spectrum[1])


plt.plot(wl_grid, tell[0, :], 'y')
plt.plot(input_spectrum[0], input_spectrum[1], 'r')
#plt.plot(wl_grid, obs[0,:], 'r')
plt.plot(wl_grid, solution)
plt.xlim([min(wl_grid), max(wl_grid)])
output_file = os.path.join(output_dir, config['file_spectrum'])
plt.savefig(output_file, bbox_inches='tight')
output_file = os.path.join(output_dir, config['file_data_out'])
np.savetxt(output_file, solution)
plt.show()
