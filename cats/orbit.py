"""
Calculate intermediary data products like
specific intensities or F and G
"""
import matplotlib.pyplot as plt
import numpy as np

import ExoOrbit

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
        transit = 0 #self.par["transit"]
        # periastron = self.par["periastron"]
        teff = self.par["teff"]

        star = ExoOrbit.Star(m_star, r_star, teff, name=star_name)
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

    def get_transit_depth(self, time):
        td = self._backend.transit_depth(time)
        return td

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
