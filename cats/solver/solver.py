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

        # Out of transit mask
        idx = area == 0

        # Profile of the observations
        time = times.mjd
        profile = np.nanmean(spectra, axis=1)
        model_profile = np.nanmean(model, axis=1)

        x = np.arange(len(profile))
        coeff = np.polyfit(x[idx], profile[idx], 5)
        norm = np.polyval(coeff, x)

        # plt.plot(profile)
        # plt.plot(norm)
        # plt.show()

        coeff = np.polyfit(x[idx], model_profile[idx], 5)
        model_norm = np.polyval(coeff, x)

        # plt.plot(model_profile)
        # plt.plot(model_norm)
        # plt.show()

        norm /= model_norm

        # TODO: Check that mu calculation, matches the observed transit
        # TODO: Why is the mu calculation wider than the observations?

        func = lambda x: profile - np.nanmean(
            (stellar - intensities * np.abs(x[:, None])) * telluric * norm[:, None],
            axis=1,
        )
        res = least_squares(func, x0=area, method="lm")
        area = gaussian_filter1d(res.x, 1)

        model = (stellar - intensities * area[:, None]) * telluric * norm[:, None]

        # plt.plot(np.nanmean(spectra, axis=1))
        # plt.plot(np.nanmean(model, axis=1))
        # plt.show()
        # img = spectra - model
        # plt.imshow(img, aspect="auto")
        # plt.show()

        # sort = np.argsort(times)
        # i = np.arange(101)[sort][51]

        # plt.plot(wavelength[i], model[i])
        # plt.plot(wavelength[i], spectra[i])
        # plt.show()

        f = (
            intensities
            * self.area_atmosphere
            / self.area_planet
            * area[:, None]
            * telluric
            * norm[:, None]
        )
        g = spectra - (stellar - intensities * area[:, None]) * telluric * norm[:, None]
        # f, g = f.to_value(1), g.to_value(1)

        # Normalize again
        # for i in tqdm(range(g.shape[1])):
        #     coeff = np.polyfit(time, g[:, i], 2)
        #     g[:, i] /= np.polyval(coeff, time)
        #     g[:, i] -= np.nanmedian(g[:, i])

        return wavelength, f, g

    def solve(
        self, times, wavelength, spectra, stellar, intensities, telluric, **kwargs
    ):
        raise NotImplementedError
