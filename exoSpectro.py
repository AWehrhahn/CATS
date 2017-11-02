"""
Use an inverse method approach to determine the planetary transit spectrum
author: Ansgar Wehrhahn (ansgar.wehrhahn@physics.uu.se)
Based on work by Erik Aaronson (Uppsala University)
"""

from os.path import exists, join
import numpy as np
import matplotlib.pyplot as plt

from read_write import read_write
from intermediary import intermediary
from solution import solution

# Step 1: Load data
# Step 2: Calculate intermediary products F and G
# Step 3: Calculate planet spectrum


def normalize(a):
    """ normalize array a along axis """
    a = np.abs(a)
    a -= np.min(a)
    return a / np.max(a)


#-------------------
# Step 1: Load data
#-------------------

print('Loading data...')
# Initialize IO and config data
io = read_write()
print('Loading orbital parameters')
par = io.load_parameters()

if exists(io.intermediary_file) and not io.renew_all:
    print('Load existing intermediary data F and G')
    obs, tell, wl_grid, F, G = io.load_intermediary()
else:
    print('Loading observation data')
    obs, wl_grid = io.load_observation(par['n_exposures'])

    print('Loading telluric data')
    tell = io.load_tellurics(wl_grid, par['n_exposures'])

    print('Loading stellar model')
    star_flux, star_data = io.load_star_model(
        wl_grid, par['fwhm'], par['width'])

    print('Loading complete')

    #--------------------
    # Step 2: Calculate intermediate products F and G
    # G = Telluric * I_atm
    # F = -Obs + Flux_star * Telluric - (R_planet/R_star)**2 * Telluric * I_planet
    #--------------------

    print('Calculate intermediary products')
    iy = intermediary(io.config)

    # Calculate the distances of the star to the planet during transit
    dt = iy.distance_transit(par)
    # radial velocities during transit, linearly distributed
    vel_b = iy.rv_star(par)  # rv of star, barycentric
    vel_p = iy.rv_planet(par)  # rv of exoplanet

    print('Calculate specific intensities blocked by planetary atmosphere')
    # TODO find more efficient way to calculate the specific intensities
    i_atm = iy.intensity_atmosphere(par, dt, star_data, n=20)
    i_atm = iy.doppler_shift(i_atm, wl_grid, vel_b)

    print('Calculate specific intensities blocked by solid planet body')
    # TODO I changed the radii at which the intensity is calculated check if that is correct or not
    i_planet = iy.intensity_planet(par, dt, star_data, n=20, m=20)
    i_planet = iy.doppler_shift(i_planet, wl_grid, vel_b)

    print('Doppler shift stellar flux')
    star_flux = iy.doppler_shift(star_flux, wl_grid, vel_b)

    print('Brightness correction parameter')
    mu = iy.brightness_correction(par, obs, star_flux, tell, i_planet, i_atm)
    obs *= mu[:, None] #to fix the axes

    print('Calculate G = Telluric * I_atmosphere')
    G = iy.calc_G(i_atm, tell)
    G = iy.doppler_shift(G, wl_grid, -vel_b - vel_p)

    print('Calculate F = -Observation + Flux * Telluric - (R_planet/R_star)**2 * I_planet * Telluric')
    F = iy.calc_F(obs, star_flux, tell, i_planet,
                  par['r_planet'], par['r_star'])
    F = iy.doppler_shift(F, wl_grid, -vel_b - vel_p)

    io.save_intermediary(obs, tell, wl_grid, F, G)

# Step 3: Calculate planet spectrum
#   - try different values for regularization parameter lambda

print('Solve minimization problem for planetary spectrum')
sn = solution()
sol = sn.solve(wl_grid, F, G, 1500)
sol = normalize(sol)

# Step 4: Profit, aka Plotting
input_spectrum = io.load_input(wl_grid)
input_spectrum = normalize(input_spectrum)

plt.plot(wl_grid, tell[0, :], 'y')
plt.plot(input_spectrum[0], input_spectrum[1], 'r')
#plt.plot(wl_grid, obs[0,:], 'r')
plt.plot(wl_grid, sol)
plt.xlim([min(wl_grid), max(wl_grid)])

# save plot
output_file = join(io.output_dir, io.config['file_spectrum'])
plt.savefig(output_file, bbox_inches='tight')
# save data
output_file = join(io.output_dir, io.config['file_data_out'])
np.savetxt(output_file, sol)
plt.show()
