import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.constants import c
from tqdm import tqdm
from scipy.sparse import diags

from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

from exoorbit.orbit import Orbit
from .least_squares import least_squares
from ..reference_frame import PlanetFrame, TelescopeFrame


class SolverBase:
    def __init__(self, detector, star, planet, **kwargs):
        self.star = star
        self.planet = planet
        self.detector = detector

        # Determine Planet Size
        if star is not None and planet is not None:
            scale_height = planet.atm_scale_height(star.teff)
            area_planet = planet.area / star.area
            area_atmosphere = np.pi * (planet.radius + scale_height) ** 2
            area_atmosphere /= star.area
            area_atmosphere -= area_planet
            self.area_planet = area_planet.to_value(u.one)
            self.area_atmosphere = area_atmosphere.to_value(u.one)

        # Set the Reference frames
        if detector is not None and star is not None:
            self.telescope_frame = TelescopeFrame(
                detector.observatory, star.coordinates
            )
        if star is not None and planet is not None:
            self.planet_frame = PlanetFrame(star, planet)

    def prepare_fg(self, times, wavelength, spectra, stellar, intensities, telluric):
        """
        Find the least-squares solution to the linear equation
        f * x - g = 0
        """

        orb = Orbit(self.star, self.planet)
        area = orb.stellar_surface_covered_by_planet(times)

        model = (stellar - intensities * area[:, None]) * telluric
        idx = np.arange(len(times))
        idx = np.concatenate((idx[:20], idx[-20:]))
        lvl = np.mean(np.nanmean(spectra[idx], axis=1) / np.nanmean(model[idx], axis=1))
        model *= lvl

        model = (stellar - intensities * area[:, None]) * telluric * lvl

        # TODO: Check that mu calculation, matches the observed transit
        # TODO: Why is the mu calculation wider than the observations?
        # plt.plot(np.nanmean(spectra, axis=1))
        # plt.plot(np.nanmean(model, axis=1))
        # plt.show()

        func = lambda x: np.nanmean(spectra, axis=1) - np.nanmean(
            (stellar - intensities * np.abs(x[:, None])) * telluric * lvl, axis=1
        )
        res = least_squares(func, x0=area, method="lm")
        area = gaussian_filter1d(res.x, 1)
        # area[:20] = 0
        # area[-20:] = 0

        model = (stellar - intensities * area[:, None]) * telluric * lvl

        # plt.plot(np.nanmean(spectra, axis=1))
        # plt.plot(np.nanmean(model, axis=1))
        # plt.show()

        # img = spectra - model
        # plt.imshow(img, aspect="auto")
        # plt.show()

        f = (
            intensities
            * self.area_atmosphere
            / self.area_planet
            * area[:, None]
            * telluric
            * lvl
        )
        g = spectra - (stellar - intensities * area[:, None]) * telluric * lvl
        f, g = f.to_value(1), g.to_value(1)

        return wavelength, f, g

    def solve(
        self, times, wavelength, spectra, stellar, intensities, telluric, **kwargs
    ):
        pass
