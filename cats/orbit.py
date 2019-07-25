"""
Calculate intermediary data products like
specific intensities or F and G
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np

import ExoOrbit

from .data_modules.dataset import dataset


class Orbit:
    """Calculates the orbital parameters of the transiting planet

    Functions
    -------
    get_phase(obs_time): {phase}

    get_radius(phase): {radius}

    get_pos(phase): {x, y, z coordinates}

    get_mu(x, y, z, angles, radii): {mu}

    maximum_phase(): {phase}
        gets the maximum phase that is still considered in transit
    rv_planet(phase): {radial velocity}

    plot(x, y, z, mode='2D'): {}
    """

    def __init__(self, configuration, parameters):
        self.configuration = configuration
        self.par = parameters

        if self.configuration is not None:
            star_name = self.configuration["_star"]
            planet_name = self.configuration["_planet"]
        else:
            star_name = ""
            planet_name = ""

        r_star = self.par["r_star"]
        m_star = self.par["m_star"]

        r_planet = self.par["r_planet"]
        m_planet = self.par["m_planet"]

        sma = self.par["sma"]
        period = self.par["period"]
        ecc = self.par["ecc"]
        inc = self.par["inc"]
        w = self.par["w"]
        transit = self.par["transit"]
        # periastron = self.par["periastron"]

        star = ExoOrbit.Star(m_star, r_star, name=star_name)
        planet = ExoOrbit.Planet(
            m_planet, r_planet, sma, period, ecc, inc, w, transit, name=planet_name
        )

        self._backend = ExoOrbit.Orbit(star, planet)

    def get_phase(self, times):
        """Calculate the orbit phase depending on the obs_time

        Parameters:
        ----------
        times : {float, np.ndarray}
            observation times in MJD
        Returns
        -------
        phase: {float, np.ndarray}
            orbital phase in radians
        """

        return self._backend.phase_angle(times)

    def get_radius(self, times):
        """calculate the radius at the given time

        this is NOT the radius in the direction of observation, but the radius in polar coordinates

        Parameters:
        ----------
        times : {float, np.ndarray}
            observation times in MJD
        Returns
        -------
        radius : {float, np.ndarray}
            radius in units of stellar radii
        """

        return self._backend.projected_radius(times)

    def get_pos(self, times):
        """Calculate the 3D position of the planet

        the coordinate system is centered in the star, x is towards the observer, z is "north", and y to the "right"

          z ^
            |
            | -¤-
            |̣_____>
            /      y
           / x

        Parameters:
        ----------
        phase : {float, np.ndarray}
            phase in radians
        Returns
        -------
        x, y, z: {float, np.ndarray}
            position in stellar radii
        """

        return self._backend.position_3D(times)

    def maximum_phase(self):
        """ The maximum phase for which the planet is still completely inside the stellar disk

        Returns
        -------
        phase : float
            maximum phase (in radians)
        """

        t2 = self._backend.second_contact()
        t3 = self._backend.third_contact()
        t = np.asarray([t2, t3])

        max_phase = self._backend.phase_angle(t)
        return max_phase

    def get_mu(self, times):
        """get mu = cos(distance to stellar center)
        cos(limb distance), where 1 is the center of the star and 0 is the outer edge

        project onto yz-plane, i.e. ignore x and calculate the distance

        Parameters:
        ----------
        x : {float, np.ndarray}
            x position in stellar radii
        y : {float, np.ndarray}
            y position in stellar radii
        z : {float, np.ndarray}
            z position in stellar radii
        Returns
        -------
        mu : float, np.ndarray
            cos(sqrt(y**2 + z**2))
        """
        return self._backend.mu(times)

    def get_rv(self, times):
        """Calculate the radial velocity of the planet at a given phase

        Parameters:
        ----------
        times : {float, np.ndarray}
            observation times
        Returns
        -------
        rv : {float, np.ndarray}
            radial velocity of the planet in km/s
        """
        v = self._backend.radial_velocity_planet(times)
        v *= 1e-3  # convert to km/s
        return v

    def plot(self, x, y, z, mode="2D"):
        """Plot the star together with planet at position x, y, z

        Parameters:
        ----------
        x : {np.ndarray}
            x-coordinate
        y : {np.ndarray}
            y-coordinate
        z : {np.ndarray}
            z-coordinate
        """

        if mode == "2D":
            c_star = plt.Circle((0, 0), 1, color="y")

            r = (self.par["r_planet"] + self.par["h_atm"]) / self.par["r_star"]
            c = [
                plt.Circle((j, k), r, color="b", alpha=0.5) for i, j, k in zip(x, y, z)
            ]

            fig, ax = plt.subplots()
            ax.add_artist(c_star)
            ax.plot(y, z, "o", color="r")
            for circle in c:
                ax.add_artist(circle)

            plt.xlabel("y")
            plt.ylabel("z")
            plt.xlim((-1.5, 1.5))
            plt.ylim((-1.5, 1.5))
            plt.show()

        if mode == "3D":
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.set_proj_type("ortho")

            # draw sphere/star
            def plot_sphere(x, y, z, radius, color="b", alpha=0.5):
                u = np.linspace(0, 2 * np.pi, 100)
                v = np.linspace(0, np.pi, 100)

                _x = x + radius * np.outer(np.cos(u), np.sin(v))
                _y = y + radius * np.outer(np.sin(u), np.sin(v))
                _z = z + radius * np.outer(np.ones(np.size(u)), np.cos(v))
                ax.plot_surface(
                    _x,
                    _y,
                    _z,
                    rstride=4,
                    cstride=4,
                    color=color,
                    linewidth=0,
                    alpha=alpha,
                )

            plot_sphere(0, 0, 0, 1, "y", 1)

            r = (self.par["r_planet"] + self.par["h_atm"]) / self.par["r_star"]
            for i, j, k in zip(x, y, z):
                ax.scatter(i, j, k, color="r")
                plot_sphere(i, j, k, r, "b")

            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.view_init(elev=0, azim=0)

            plt.show()


def profile_exponential(par, h):
    scale_height = par["atm_scale_height"] / par["r_star"] * 10000
    return np.exp(-h / scale_height)


def profile_solid(par, h):
    r = np.min(h, axis=1)
    ha = np.max(h)
    L = np.sqrt(ha ** 2 - r ** 2)
    return np.repeat(0.5 / L, h.shape[1]).reshape(h.shape)


def atmosphere_profile(
    par,
    phase,
    intensity,
    r0,
    h,
    profile=profile_exponential,
    n_radii=11,
    n_angle=12,
    n_depth=13,
):
    orb = orbit(par)
    _, y, z = orb.get_pos(phase)

    r = np.linspace(r0, h, n_radii, endpoint=False)
    theta = np.linspace(0, 2 * np.pi, n_angle, endpoint=False)
    L = np.sqrt(h ** 2 - r ** 2)
    x = np.array([np.linspace(0, l, n_depth) for l in L])
    x_dash = np.sqrt(r[:, None] ** 2 + x ** 2)

    rho_i = 2 * simps(profile(par, x_dash - r0), x, axis=1)
    rho_i = np.nan_to_num(rho_i, copy=False)

    mu = orb.get_mu(_, y, z, angles=theta, radii=r)
    I = interpolate_intensity(mu, intensity)

    # TODO optimize this multiplication?
    tmp = simps(
        I * mu[..., None] * rho_i[None, :, None, None] * r[None, :, None, None],
        r,
        axis=1,
    )
    res = simps(tmp, theta, axis=1) / np.pi  # TODO normalize to area of star
    return res


def test_profile(par, phase, intensity, r0, *args, **kwargs):
    orb = orbit(par)
    _, y, z = orb.get_pos(phase)
    H = par["atm_scale_height"]
    r0 = r0 * par["r_star"]

    mu = lambda r, theta: orb.get_mu(0, y, z, angles=theta, radii=r)
    I = lambda r, theta: interpolate_intensity(
        orb.get_mu(0, y, z, angles=theta, radii=r), intensity
    )
    rho = lambda r, x: np.exp(-(np.sqrt(r ** 2 - x ** 2) - r0) / H)

    func = lambda x, theta, r: mu(r, theta)[0] * I(r, theta)[0, 0] * rho(r, x) * r
    res = tplquad(
        func,
        r0,
        np.inf,
        lambda x: 0,
        lambda x: 2 * np.pi,
        lambda x, theta: -np.inf,
        lambda x, theta: +np.inf,
    )
    return res


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
    # TODO can I optimize this?
    values = i.values.swapaxes(0, 1)
    return interp1d(
        i.keys(),
        values,
        kind="zero",
        axis=0,
        bounds_error=False,
        fill_value=(0, values[1]),
    )(mu)


def calc_intensity(
    par,
    phase,
    intensity,
    min_radius,
    max_radius,
    n_radii,
    n_angle,
    spacing="equidistant",
):
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
    if spacing in ["e", "equidistant"]:
        # equidistant spacing
        radii = np.linspace(min_radius, max_radius, n_radii, endpoint=True)
        # No endpoint means no overlap -> no preference (but really thats just a small difference)
        angles = np.linspace(0, 2 * np.pi, n_angle, endpoint=False)
    if spacing in ["r", "random", "mc"]:
        # random spacing (Monte-Carlo)
        radii = (
            np.random.random_sample(n_radii) * (max_radius - min_radius) + min_radius
        )
        angles = np.random.random_sample(n_angle) * 2 * np.pi

    # Step 2: Calculate mu, distances from the stellar center
    radii /= par["r_star"]
    mu = calc_mu(par, phase, angles=angles, radii=radii)
    # -1 is out of bounds, which will be filled with 0 intensity
    mu[np.isnan(mu)] = -1

    # Step 3: Average specific intensity, outer points weight more, as the area is larger
    # flux = integrate intensity(mu) * mu dmu = Area * average(Intensity * mu)
    intens = interpolate_intensity(mu, intensity)
    intens = np.average(intens, axis=2)
    intens = np.average(intens, axis=1, weights=radii)
    return intens


def specific_intensities(par, phase, intensity, n_radii=11, n_angle=7, mode="fast"):
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

    if mode == "profile":
        r0 = par["r_planet"] / par["r_star"]
        h = r0 + par["h_atm"] / par["r_star"]

        planet = atmosphere_profile(
            par, phase, intensity.flux, 0, h, profile_solid, n_radii[0], n_angle[0]
        )
        atm = atmosphere_profile(
            par,
            phase,
            intensity.flux,
            r0,
            h,
            profile_exponential,
            n_radii[1],
            n_angle[1],
        )

        ds_planet = dataset(intensity.wl, planet)
        ds_atm = dataset(intensity.wl, atm)
        return ds_planet, ds_atm

    if mode == "precise":
        # from r=0 to r = r_planet + r_atmosphere
        inner = 0
        outer = par["r_planet"] + par["h_atm"]
        i_planet = calc_intensity(
            par, phase, intensity.flux, inner, outer, n_radii[0], n_angle[0]
        )

        # from r=r_planet to r=r_planet+r_atmosphere
        inner = par["r_planet"]
        outer = par["r_planet"] + par["h_atm"]
        i_atm = calc_intensity(
            par, phase, intensity.flux, inner, outer, n_radii[1], n_angle[1]
        )
        ds_planet = dataset(intensity.wl, i_planet)
        ds_atm = dataset(intensity.wl, i_atm)
        return ds_planet, ds_atm

    if mode == "fast":
        # Alternative version that only uses the center of the planet
        # Faster but less precise (significantly?)
        mu = calc_mu(par, phase)
        _flux = interpolate_intensity(mu, intensity.flux)
        ds = dataset(intensity.wl, np.copy(_flux))
        ds2 = dataset(intensity.wl, np.copy(_flux))
        return ds, ds2
