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

    def distance_transit(self):
        """
        Calculate the distances from the centre of the stellar disk
        to the centre of the planet along the transit
        self.par: Orbital self.parameters
        """

        r"""
            |\
          d | \
            |  \ r_s
            |__i\
            td
        """

        i = np.deg2rad(self.par['inc'])
        # distance from the ecliptic; i = 90 degree : Edge on orbit, transit along the ecliptic
        d = np.cos(i) * self.par['sma']
        # half of the total distance traveled over the star
        total_distance = np.sqrt(self.par['r_star']**2 - d**2)

        # split the total traveled distance into n equally spaced self.parts, where n is the number of exposures
        distances = np.linspace(
            0, 2 * (total_distance - self.par['r_planet']) / self.par['r_star'], self.par['n_exposures'])

        # relative to the center of transit
        distances = distances - np.mean(distances)

        # distance from centre of stellar disk to centre of planet
        distance_center = np.sqrt((d / self.par['r_star'])**2 + (distances)**2)

        return distance_center

    def calc_distances(self, sample_radius, d, n):
        """
        Calculate distances of points in the atmosphere/planet to the center of the stallar disk
        self.par: Orbital self.parameters
        sample_radius: radius from the planet center to sample 
        d: distances from stellar center along the orbit
        n: number of points in the atmosphere
        """
        if isinstance(sample_radius, np.ndarray):
            return np.array([self.calc_distances(s, d, n) for s in sample_radius])
        # size of the planet (with atmosphere) in stellar radii
        # sample radius = (self.par['r_planet'] + self.par['h_atm'] / 2)
        r = sample_radius / self.par['r_star']

        # whole circle split into n self.parts
        phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
        # law of cosines, distances to different points in the atmosphere
        x = np.sqrt(d**2 + r**2 - 2 * d * r * np.cos(phi))

        # if x > r_star then return r_star
        # Use the where keyword to avoid unnecessary calculations
        x[x > 1] = 1
        return x

    def intensity_interpolation(self, dt, i):
        """ Interpolate values of distances between starI files i """
        d = np.sqrt(1 - dt**2)  # cos(arcsin(dt))
        return interp1d(i.keys().values, i.values, kind=self.config['interpolation_method'], fill_value='extrapolate', bounds_error=False)(d)

    def calc_specific_intensity(self, dt, radii, star_data, n=20):
        """
        Calculate specific intensities blocked by solid planet body
        self.par: Orbital self.parameters
        dt: planet-star distances during transit
        radii: distances from the center of the planet (in km) to sample
        star_data: stellar intensity data
        n: number of angles to sample
        """
        if isinstance(dt, np.ndarray):
            return np.array([self.calc_specific_intensity(k, radii, star_data, n=n) for k in dt])

        distance_planet = self.calc_distances(radii, dt, n)
        # outer radii should have larger weight in the average due to larger area contribution
        # weights scale linearly with radius because dA = 2*pi*r*dr, if we assume dr << r

        i = np.array([self.intensity_interpolation(
            distance_planet[j, :], star_data) for j in range(len(radii))])
        i = np.mean(i, axis=2)
        weight = (2 * np.arange(len(radii)) + 1) / len(radii)**2
        i = np.sum(i * weight[:, None], axis=0)
        return i

    def intensity_atmosphere(self, dt, star_data, n=20):
        """
        Calculate the specific intensties blocked by the planetary atmosphere
        self.par: self.paramters
        dt: planet-star distances during transit
        star_data: star intensity data
        n: number of points in the atmosphere
        """
        # Sample the center of the atmosphere
        sample_radius = np.array(
            [self.par['r_planet'] + self.par['h_atm'] / 2])
        return self.calc_specific_intensity(dt, sample_radius, star_data, n=n)

    def intensity_planet(self, dt, star_data, n=20, m=20):
        """
        Calculate specific intensities blocked by solid planet body
        self.par: Orbital self.parameters
        dt: planet-star distances during transit
        star_data: stellar intensity data
        n: number of angles to sample
        m: number of radii to sample
        """
        # various distances from the centre of the planet sampled here
        radii = np.linspace(1 + 0.25 / m, 1.25, m) * \
            (self.par['r_planet'] + self.par['h_atm'])
        return self.calc_specific_intensity(dt, radii, star_data, n=n)

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

    def brightness_correction(self, obs, star_flux, tell, I_planet, I_atm):
        """ calculate the brightness correction factor """
        res = np.zeros((2, 200, self.par['n_exposures']))

        antinorm = 1.01 - np.arange(200) / 3000
        #residual = antinorm * obs - (star_flux-I_planet*self.par['A_planet']) * tell
        for an in range(200):
            residual = (antinorm[an] * obs - (star_flux - I_planet *
                                              (self.par['r_planet'] / self.par['r_star'])**2) * tell)

            for i in range(self.par['n_exposures']):
                # This return non normalized cross correlation, but thats ok, because we are just interested in the minumum anyways
                result1 = np.correlate(residual[i], star_flux[i])
                result2 = np.correlate(residual[i], tell[i])

                res[0, an, i] = antinorm[an]
                res[1, an, i] = np.abs(result1 * result2)

        norm = np.zeros(self.par['n_exposures'])
        for i in range(self.par['n_exposures']):
            norm[i] = res[0, res[1, :, i] == min(res[1, :, i]), i]
        return norm

    def calc_F(self, obs, flux, tell, i_planet, r_planet, r_star):
        """ Calculate the intermediary product F, which is independant of planet spectrum P """
        return - obs + flux * tell - (r_planet / r_star)**2 * i_planet * tell

    def calc_G(self, i_atm, tell):
        """ Calculate the intermediary product G, which is dependant on planet spectrum P """
        return i_atm * tell

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
