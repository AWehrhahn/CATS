import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad, trapz, simps, tplquad


from ..orbit import Orbit as orbit_calculator
from .data_interface import data_intensities as di
from .dataset import dataset

# TODO: replace Orbit with ExoOrbit

class data_intensities(di):
    def __init__(self, configuration):
        self.parameters = {}
        self.orbit = None
        super().__init__(configuration)

    # Call this the first time
    def init(self, **data):
        self.parameters = data["parameters"]
        self.orbit = orbit_calculator(self.configuration, self.parameters)

    def profile_exponential(self, h):
        atm_scale_height = self.parameters["atm_scale_height"]
        r_star = self.parameters["r_star"]
        scale_height = atm_scale_height / r_star * 10000
        return np.exp(-h / scale_height)

    def profile_solid(self, h):
        r = np.min(h, axis=1)
        ha = np.max(h)
        L = np.sqrt(ha**2 - r**2)
        return np.repeat(0.5 / L, h.shape[1]).reshape(h.shape)


    def atmosphere_profile(self, phase, intensity, r0, h, profile="exponential", n_radii=11, n_angle=12, n_depth=13):
        if profile == "exponential":
            profile = self.profile_exponential
        elif profile == "solid":
            profile = self.profile_solid
        else:
            raise ValueError

        _, y, z = self.orbit.get_pos(phase)

        r = np.linspace(r0, h, n_radii, endpoint=False)
        theta = np.linspace(0, 2 * np.pi, n_angle, endpoint=False)
        L = np.sqrt(h**2 - r**2)
        x = np.array([np.linspace(0, l, n_depth) for l in L])
        x_dash = np.sqrt(r[:, None]**2 + x**2)

        rho_i = 2 * simps(profile(x_dash - r0), x, axis=1)
        rho_i = np.nan_to_num(rho_i, copy=False)

        mu = self.orbit.get_mu(_, y, z, angles=theta, radii=r)
        I = self._class__.interpolate_intensity(mu, intensity)

        # TODO optimize this multiplication?
        tmp = simps(I * mu[..., None] * rho_i[None, :, None, None]
                    * r[None, :, None, None], r, axis=1)
        res = simps(tmp, theta, axis=1) / np.pi  # TODO normalize to area of star
        return res


    def test_profile(self, phase, intensity, r0, *args, **kwargs):
        _, y, z = self.orbit.get_pos(phase)
        H = self.parameters['atm_scale_height']
        r0 = r0 * self.parameters['r_star']

        mu = lambda r, theta: self.orbit.get_mu(0, y, z, angles=theta, radii=r)
        I = lambda r, theta: self.__class__.interpolate_intensity(self.orbit.get_mu(0, y, z, angles=theta, radii=r), intensity)
        rho = lambda r, x: np.exp(-(np.sqrt(r**2-x**2)-r0)/H)

        func = lambda x, theta, r:  mu(r, theta)[0] * I(r, theta)[0, 0] * rho(r, x) * r
        res = tplquad(func, r0, np.inf, lambda x: 0, lambda x: 2*np.pi, lambda x, theta: -np.inf, lambda x, theta: +np.inf)
        return res

    @staticmethod
    def interpolate_intensity(mu, i):
        """ Interpolate the stellar intensity for given limb distance mu

        use linear interpolation, because it is much faster

        Parameters:
        ----------
        mu : {float, np.ndarray}
            cos(limb distance), i.e. 1 is the center of the star, 0 is the outer edge
        i : {pd.DataFrame}
            specific intensities
        Returns
        -------
        intensity : np.ndarray
            interpolated intensity
        """

        values = i.values.swapaxes(0, 1)
        keys = np.asarray(i.keys())
        flux = interp1d(keys, values, kind='linear', axis=0, bounds_error=False, fill_value=(values[0], values[-1]))(mu)
        flux[mu < 0, :] = 0
        return flux


    def calc_intensity(self, phase, intensity, min_radius, max_radius, n_radii, n_angle, spacing='equidistant'):
        """Calculate the average specific intensity in a given radius range around the planet center

        Parameters:
        ----------
        par : {dict}
            stellar and planetary parameters
        phase : {float, np.ndarray}
            orbital phase of the planet (in radians)
        intensity : {datset}
            specific intensities
        min_radius : {float}
            minimum radius from the planet center to use for calculations (in km)
        max_radius : {float}
            maximum radius from planet center to use for calculations (im km)
        n_radii : {int}
            number of radii to sample
        n_angle : {int}
            number of angles to sample
        spacing : {'equidistant', 'random'}
            wether to use equidistant sampling points or a random distribution (default is 'equidistant')
        Returns
        -------
        intensity : np.ndarray
            specific intensities
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

        # Step 2: Calculate mu, distances from the stellar center
        radii /= self.parameters['r_star']
        mu = self.calc_mu(phase, angles=angles, radii=radii)
        # -1 is out of bounds, which will be filled with 0 intensity
        mu[np.isnan(mu)] = -1

        # Step 3: Average specific intensity, outer points weight more, as the area is larger
        # flux = integrate intensity(mu) * mu dmu = Area * average(Intensity * mu)
        intens = self.__class__.interpolate_intensity(mu, intensity)
        intens = np.average(intens, axis=2)
        intens = np.average(intens, axis=1, weights=radii)
        return intens


    def calc_mu(self, time, angles=None, radii=None):
        """calculate the distance from the center of the planet to the center of the star as seen from earth

        Parameters:
        ----------
        par : {dict}
            stellar and planetary parameters
        phase : {float, np.ndarray}
            orbital phase of the planet
        angles: {np.ndarray}, optional
            set of angles to sample around the center of the planet (in radians)
        radii: {np.ndarray}, optional
            set of radii to sample around the center of the planet (in km)
        Returns
        -------
        mu : {float, np.ndarray}
            cos(limb distance), where 0 is the center of the star and 1 is the outer edge
        """

        return self.orbit.get_mu(time)

    def get_specifics(self, time, intensity, n_radii=11, n_angle=7, mode='fast'):
        """Calculate the specific intensities of the star covered by planet and atmosphere, and only atmosphere respectively,
        over the different phases of transit

        TODO: are the precise calculation worth it ???
        TODO: Allow user to specify different n_radii and n_angle for i_planet and i_atm

        Parameters:
        ----------
        par : {dict}
            stellar and planetary parameters
        phase : {float, np.ndarray}
            orbital phase in radians
        intensity : {dataset}
            specific intensities
        n_radii : {int}, optional
            number of radii to sample (the default is 11)
        n_angle : {int}, optional
            number of angles to sample (the default is 7)
        mode : {'precise', 'fast'}
            in precise mode, various radii and angle in the atmosphere and body of the planet are sampled
            inb fast mode, use only the center of the planet
        Returns
        -------
        i_planet, i_atm : dataset, dataset
            specific intensities blocked by the planet body and the atmosphere
        """

        if isinstance(n_radii, (float, int)):
            n_radii = (n_radii, n_radii)
        if isinstance(n_angle, (float, int)):
            n_angle = (n_angle, n_angle)

        if mode == 'profile':
            r0 = self.parameters['r_planet'] / self.parameters['r_star']
            h = r0 + self.parameters['h_atm'] / self.parameters['r_star']

            planet = self.atmosphere_profile(time, intensity.flux, 0, h, "solid", n_radii[0], n_angle[0])
            atm = self.atmosphere_profile(time, intensity.flux, r0, h, "exponential", n_radii[1], n_angle[1])

            ds_planet = dataset(intensity.wl, planet)
            ds_atm = dataset(intensity.wl, atm)
            return ds_planet, ds_atm
        elif mode == 'precise':
            # from r=0 to r = r_planet + r_atmosphere
            inner = 0
            outer = self.parameters['r_planet'] + self.parameters['h_atm']
            i_planet = self.calc_intensity(time, intensity.data, inner, outer, n_radii[0], n_angle[0])

            # from r=r_planet to r=r_planet+r_atmosphere
            inner = self.parameters['r_planet']
            outer = self.parameters['r_planet'] + self.parameters['h_atm']
            i_atm = self.calc_intensity(time, intensity.data,
                                inner, outer, n_radii[1], n_angle[1])
            ds_planet = dataset(intensity.wave, i_planet)
            ds_atm = dataset(intensity.wave, i_atm)
            return ds_planet, ds_atm
        elif mode == 'fast':
            # Alternative version that only uses the center of the planet
            # Faster but less precise (significantly?)
            mu = self.calc_mu(time)
            mu[mu == -1] = 0
            k2 = self.parameters["A_planet"].value
            depth = self.orbit.get_transit_depth(time) 
            depth /= k2
            flux = self.__class__.interpolate_intensity(mu, intensity.data)
            flux *= depth[:, None] 
            ds = dataset(intensity.wave, np.copy(flux))
            ds2 = dataset(intensity.wave, np.copy(flux))
            return ds, ds2
        else:
            raise ValueError


    def load_intensities(self, **data):
        """ Load specific intensity data, for this method

        Returns
        -------
        ds : dataset
            With wave = wavelength
            and flux = pandas dataframe with columns mu and intensities in rows
        """
        raise NotImplementedError

    def get_intensities(self, **data):
        self.init(**data)
        # phases = data["observations"].phase
        times = data["observations"].time
        intensity = self.load_intensities(**data)
        i_core, i_atmo = self.get_specifics(times, intensity)
        return i_core, i_atmo
