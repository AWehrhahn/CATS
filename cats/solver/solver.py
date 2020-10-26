import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.constants import c
from tqdm import tqdm
from scipy.sparse import diags

from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

from exoorbit.orbit import Orbit
from ..least_squares.least_squares import least_squares
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
        model = stellar * telluric

        # Normalize the profile of the observations
        profile = np.nanmean(spectra, axis=1)
        model_profile = np.nanmean(model, axis=1)
        norm = profile / model_profile

        model = stellar * telluric * norm[:, None]
        diff = spectra - model

        f = -(
            np.nan_to_num(intensities)
            * self.area_atmosphere
            / self.area_planet
            * area[:, None]
            * telluric
            * norm[:, None]
        )
        # g = spectra - (stellar - intensities * area[:, None]) * telluric * norm[:, None]
        g = spectra - stellar * telluric * norm[:, None]

        return wavelength, f, g

    def solve(
        self, times, wavelength, spectra, stellar, intensities, telluric, **kwargs
    ):
        raise NotImplementedError
