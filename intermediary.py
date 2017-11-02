"""
Calculate intermediary data products like
specific intensities or F and G
"""

import numpy as np
from scipy.interpolate import interp1d


class intermediary:
    """ Wrapper class for various intermediary data product functions """

    def __init__(self, config):
        self.config = config

    def distance_transit(self, par):
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

    def calc_distances(self, par, sample_radius, d, n):
        """
        Calculate distances of points in the atmosphere/planet to the center of the stallar disk
        par: Orbital parameters
        sample_radius: radius from the planet center to sample 
        d: distances from stellar center along the orbit
        n: number of points in the atmosphere
        """
        if isinstance(sample_radius, np.ndarray):
            return np.array([self.calc_distances(par, s, d, n) for s in sample_radius])
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

    def intensity_interpolation(self, dt, i):
        """ Interpolate values of distances between starI files i """
        d = np.sqrt(1 - dt**2)  # cos(arcsin(dt))
        return interp1d(i.keys().values, i.values, kind=self.config['interpolation_method'], fill_value='extrapolate', bounds_error=False)(d)

    def calc_specific_intensity(self, par, dt, radii, star_data, n=20):
        """
        Calculate specific intensities blocked by solid planet body
        par: Orbital parameters
        dt: planet-star distances during transit
        radii: distances from the center of the planet (in km) to sample
        star_data: stellar intensity data
        n: number of angles to sample
        """
        if isinstance(dt, np.ndarray):
            return np.array([self.calc_specific_intensity(par, k, radii, star_data, n=n) for k in dt])

        distance_planet = self.calc_distances(par, radii, dt, n)
        # outer radii should have larger weight in the average due to larger area contribution
        # weights scale linearly with radius because dA = 2*pi*r*dr, if we assume dr << r

        i = np.array([self.intensity_interpolation(
            distance_planet[j, :], star_data) for j in range(len(radii))])
        i = np.mean(i, axis=2)
        weight = (2 * np.arange(len(radii)) + 1) / len(radii)**2
        i = np.sum(i * weight[:, None], axis=0)
        return i

    def intensity_atmosphere(self, par, dt, star_data, n=20):
        """
        Calculate the specific intensties blocked by the planetary atmosphere
        par: paramters
        dt: planet-star distances during transit
        star_data: star intensity data
        n: number of points in the atmosphere
        """
        # Sample the center of the atmosphere
        sample_radius = np.array([par['r_planet'] + par['h_atm'] / 2])
        return self.calc_specific_intensity(par, dt, sample_radius, star_data, n=n)

    def intensity_planet(self, par, dt, star_data, n=20, m=20):
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
        return self.calc_specific_intensity(par, dt, radii, star_data, n=n)

    def doppler_shift(self, spectrum, wl, vel):
        """ Shift spectrum by velocity vel """
        if isinstance(vel, np.ndarray) and isinstance(spectrum, np.ndarray):
            if spectrum.ndim > 1:
                return np.array([self.doppler_shift(spectrum[k], wl, vel[k]) for k in range(len(vel))])
            else:
                return np.array([self.doppler_shift(spectrum, wl, vel[k]) for k in range(len(vel))])
        c0 = 299792  # speed of light in km/s
        # new shifted wavelength grid
        wl_doppler = wl * (1 + vel / c0)
        return interp1d(wl_doppler, spectrum, kind=self.config['interpolation_method'], fill_value=0, bounds_error=False)(wl)

    def rv_star(self, par):
        """ linearly distribute radial velocities during transit """
        return np.linspace(0, par['rv_end'] - par['rv_start'], par['n_exposures']) + par['rv_start']

    def rv_planet(self, par):
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

    # TODO clean up and optimization
    def brightness_correction(self, par, obs, star_flux, tell, I_planet, I_atm):
        """ calculate the brightness correction factor """
        res = np.zeros((2, 200, par['n_exposures']))

        antinorm = 1.01 - np.arange(200) / 3000
        #residual = antinorm * obs - (star_flux-I_planet*par['A_planet']) * tell
        for an in range(200):
            residual = (antinorm[an] * obs - (star_flux - I_planet *
                                              (par['r_planet'] / par['r_star'])**2) * tell)

            for i in range(par['n_exposures']):
                # This return non normalized cross correlation, but thats ok, because we are just interested in the minumum anyways
                result1 = np.correlate(residual[i], star_flux[i])
                result2 = np.correlate(residual[i], tell[i])

                res[0, an, i] = antinorm[an]
                res[1, an, i] = np.abs(result1 * result2)

        norm = np.zeros(par['n_exposures'])
        for i in range(par['n_exposures']):
            norm[i] = res[0, res[1, :, i] == min(res[1, :, i]), i]
        return norm

    def calc_F(self, obs, flux, tell, i_planet, r_planet, r_star):
        """ Calculate the intermediary product F, which is independant of planet spectrum P """
        return  - obs + flux * tell - (r_planet / r_star)**2 * i_planet * tell

    def calc_G(self, i_atm, tell):
        """ Calculate the intermediary product G, which is dependant on planet spectrum P """
        return i_atm * tell
