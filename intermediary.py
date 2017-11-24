"""
Calculate intermediary data products like
specific intensities or F and G
"""

import os.path
import subprocess
import numpy as np
from scipy.interpolate import interp1d


class intermediary:
    """ Wrapper class for various intermediary data product functions """

    def __init__(self, config, par, dtype=np.float):
        self.config = config
        self.par = par
        self.dtype = dtype

    def doppler_shift(self, spectrum, wl, vel):
        """ Shift spectrum by velocity vel """
        c0 = 299792  # speed of light in km/s
        # new shifted wavelength grid
        doppler = 1 - vel / c0
        wl_doppler = wl[None, :] * doppler[:, None]
        return np.interp(wl_doppler, wl, spectrum)
        # return interp1d(wl, spectrum, kind=self.config['interpolation_method'], fill_value=0, bounds_error=False)(wl_doppler)

    def rv_star(self):
        """ linearly distribute radial velocities during transit """
        return np.linspace(self.par['rv_start'], self.par['rv_end'], self.par['n_exposures'])

    def rv_planet(self, phases):
        """ calculate radial velocities of the planet along the orbit """
        # Orbital speed
        v_orbit = self.par['sma'] * \
            np.sin(self.par['inc']) * 2 * np.pi / self.par['period_s']
        # Modulate with phase
        return v_orbit * np.sin(phases)

    def fit_tellurics(self, verbose=False):
        """ fit tellurics using molecfit """
        mfit = os.path.join(
            self.config['intermediary_dir'], self.config['file_molecfit'])
        molecfit = os.path.expanduser(self.config['path_molecfit'])
        sp = subprocess.Popen([molecfit, mfit], stdout=subprocess.PIPE)
        if verbose:
            for line in iter(sp.stdout.readline, ''):
                print(line.decode('utf-8').rstrip())
                if line.decode('utf-8') == '':
                    break
            sp.stdout.close()
        sp.wait()

    def interpolate_intensity(self, mu, i):
        """ interpolate the stellar intensity for given limb distance mu """
        return interp1d(i.keys().values, i.values, kind=self.config['interpolation_method'], fill_value=0, bounds_error=False, copy=False)(mu).swapaxes(0, 1)

    def calc_mu(self, phase):
        """ calculate the distance from the center of the planet to the center of the star as seen from earth """
        """
        distance = self.par['sma'] / self.par['r_star'] * \
            np.sqrt(np.cos(self.par['inc'])**2 +
                    np.sin(self.par['inc'])**2 * np.sin(phase)**2)
        """
        return np.sqrt(1 - (self.par['sma'] / self.par['r_star'])**2 * (np.cos(self.par['inc'])**2 + np.sin(self.par['inc'])**2 * np.sin(phase)**2))

    def calc_intensity(self, phase, intensity, min_radius, max_radius, n_radii, n_angle, spacing='equidistant'):
        """
        Calculate the average specific intensity in a given radius range around the planet center
        phase: Phase (in radians) of the planetary transit, with 0 at transit center
        intensity: specific intensity profile of the host star, a pandas DataFrame with keys from 0.0 to 1.0
        min_radius: minimum radius (in km) to sample
        max_radius: maximum radius (in km) to sample
        n_radii: number of radius points to sample
        n_angle: number of angles to sample
        spacing: how to space the samples, 'equidistant' means linear spacing between points, 'random' places them at random positions
        """
        # Step 1: Calculate sampling positions in the given radii
        if spacing in ['e', 'equidistant']:
            # equidistant spacing
            radii = np.linspace(min_radius, max_radius, n_radii, endpoint=True)
            # No endpoint means no overlap -> no preference (but really thats just a small difference)
            angles = np.linspace(0, 2 * np.pi, n_angle, endpoint=False)
        if spacing in ['r', 'random', 'mc']:
            # random spacing (Monte-Carlo)
            radii = np.random.random_sample(
                n_radii) * (max_radius - min_radius) + min_radius
            angles = np.random.random_sample(n_angle) * 2 * np.pi
        # Step 2: Calculate d_x and d_y, distances from the stellar center
        d_x = self.par['sma'] / self.par['r_star'] * \
            np.sin(self.par['inc']) * np.sin(phase)
        d_x = d_x[:, None, None] + \
            (radii[:, None] * np.cos(angles)[None, :])[None, :, :]
        d_y = self.par['sma'] / self.par['r_star'] * \
            np.cos(self.par['inc']) + radii[:, None] * np.sin(angles)[None, :]
        # mu = sqrt(1 - d**2)
        mu = np.sqrt(1 - (d_x**2 + d_y[None, :, :]**2))
        # Step 3: Average specific intensity, outer points weight more, as the area is larger
        intens = self.interpolate_intensity(mu, intensity)
        intens = np.average(intens, axis=3)
        intens = np.average(intens, axis=2, weights=radii)
        return intens

    def maximum_phase(self):
        """ The maximum phase for which the planet is still completely inside the stellar disk """
        # This is the inverse of calc_mu(maximum_phase()) = 1.0
        return np.arcsin(np.sqrt(((self.par['r_star'] - self.par['r_planet'] - self.par['h_atm']) / (
            self.par['sma'] * np.sin(self.par['inc'])))**2 - np.tan(self.par['inc'])**-2))

    def specific_intensities(self, phase, intensity, n_radii=11, n_angle=7, mode='precise'):
        """
        Calculate the specific intensities of the star covered by planet and atmosphere, and only atmosphere respectively,
        over the different phases of transit
        phase: phases (in radians) of the transit, with 0 at transit center
        intensity: specific intensity profile of the host star, a pandas DataFrame with keys from 0.0 to 1.0
        n_radii: number of radii to sample, if tuple use n_radii[0] for i_planet and n_radii[1] for i_atm
        n_angle: number of angles to sample, if tuple use n_angle[0] for i_planet and n_angle[1] for i_atm
        mode: fast or precise, fast ignores the planetary disk and only uses the center of the planet, precise uses sample positions inside the radii to determine the average intensity
        """
        # Allow user to specify different n_radii and n_angle for i_planet and i_atm
        if isinstance(n_radii, (float, int)):
            n_radii = (n_radii, n_radii)
        if isinstance(n_angle, (float, int)):
            n_angle = (n_angle, n_angle)

        if mode == 'precise':
            # from r=0 to r = r_planet + r_atmosphere
            i_planet = self.calc_intensity(
                phase, intensity, 0, (self.par['r_planet'] + self.par['h_atm']) / self.par['r_star'], n_radii[0], n_angle[0])
            # from r=r_planet to r=r_planet+r_atmosphere
            i_atm = self.calc_intensity(
                phase, intensity, self.par['r_planet'] / self.par['r_star'], (self.par['r_planet'] + self.par['h_atm']) / self.par['r_star'], n_radii[1], n_angle[1])
            return i_planet, i_atm
        if mode == 'fast':
            # Alternative version that only uses the center of the planet
            # Faster but less precise (significantly?)
            mu = self.calc_mu(phase)
            intensity = self.interpolate_intensity(mu, intensity)
            return intensity, intensity

    def create_bad_pixel_map(self, obs):
        """ Create a map of all bad pixels from the given set of observations """
        return np.all(obs == 0, axis=0) | np.all(obs == 1, axis=0)
