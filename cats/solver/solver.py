import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.constants import c
from tqdm import tqdm
from scipy.sparse import diags

from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

from ..pysysrem.sysrem import sysrem

from exoorbit.orbit import Orbit
from ..least_squares.least_squares import least_squares
from ..reference_frame import PlanetFrame, TelescopeFrame


class SolverBase:
    def __init__(self, detector, star, planet, n_sysrem=None, **kwargs):
        self.star = star
        self.planet = planet
        self.detector = detector
        self.n_sysrem = n_sysrem

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

    def prepare_fg(
        self, times, wavelength, spectra, stellar, intensities, telluric, area=None
    ):
        """
        Find the least-squares solution to the linear equation
        f * x - g = 0
        """

        if area is None:
            orb = Orbit(self.star, self.planet)
            area = orb.stellar_surface_covered_by_planet(times)

        model = stellar * telluric

        # Normalize the profile of the observations
        profile = np.nanmean(spectra, axis=1)
        model_profile = np.nanmean(model, axis=1)
        norm = profile / model_profile

        # Normalize the spectrum
        # model = stellar * telluric * norm[:, None]
        # profile = np.median(spectra, axis=0)
        # model_profile = np.median(model, axis=0)

        # nm = np.nanmedian(profile / model_profile)
        # norm *= nm

        # model = stellar * telluric * norm[:, None]
        # diff = spectra - model

        # model = np.nanmedian(spectra, axis=0)

        # f = -(
        #     # np.nan_to_num(intensities) *
        #     self.area_atmosphere
        #     / self.area_planet
        #     * area[:, None]
        #     # * np.nan_to_num(telluric, nan=1)
        #     * norm[:, None]
        # )
        # f = np.nan_to_num(intensities) * np.nan_to_num(telluric, nan=1) * norm[:, None]
        area *= self.area_atmosphere / self.area_planet
        f = -np.nan_to_num(intensities, nan=1) * area[:, None]
        if hasattr(f, "to_value"):
            f = f.to_value(1)

        # g = spectra - stellar * telluric * norm[:, None]
        # if self.n_sysrem is not None:
        #     g = sysrem(g, self.n_sysrem)

        g = spectra
        if self.n_sysrem is not None:
            # Use SVD directly instead of Sysrem
            g = sysrem(spectra, self.n_sysrem)
            # u, s, vh = np.linalg.svd(spectra, full_matrices=False)
            # s[: self.n_sysrem] = 0
            # s[80:] = 0
            # ic = (u * s) @ vh
            # g = ic
        else:
            # g = spectra - stellar * telluric * norm[:, None]
            gen = np.random.default_rng()
            tmp = sysrem(spectra, 5)
            g = gen.normal(
                loc=np.nanmean(tmp), scale=np.nanstd(tmp), size=spectra.shape
            )
            # g *= np.nanstd()  # std of random is 1 (in theory)

        # norm = np.nanstd(g, axis=0)
        # f /= norm
        # g /= norm

        # plt.imshow(g, aspect="auto", origin="lower")
        # plt.xlabel("Wavelength")
        # plt.ylabel("Time")
        # plt.title(f"N_Sysrem: {self.n_sysrem}")
        # plt.savefig(f"spectra_sysrem_{self.n_sysrem}.png")

        return wavelength, f, g

    def solve(
        self, times, wavelength, spectra, stellar, intensities, telluric, **kwargs
    ):
        raise NotImplementedError
