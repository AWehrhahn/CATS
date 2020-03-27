import numpy as np
from astropy import units as u
from astropy.constants import c

from ..reference_frame import TelescopeFrame, PlanetFrame


class SolverBase:
    def __init__(self, detector, star, planet, **kwargs):
        self.star = star
        self.planet = planet
        self.detector = detector

        # Determine Planet Size
        area_planet = planet.area / star.area
        area_atmosphere = np.pi * (planet.radius + planet.atm_scale_height) ** 2
        area_atmosphere /= star.area
        self.area_planet = area_planet.to_value(u.one)
        self.area_atmosphere = area_atmosphere.to_value(u.one)

        # Set the Reference frames
        self.telescope_frame = TelescopeFrame(detector.observatory, star.coordinates)
        self.planet_frame = PlanetFrame(star, planet)

    def prepare_fg(self, times, wavelength, spectra, stellar, intensities, telluric):
        """
        Find the least-squares solution to the linear equation
        f * x - g = 0
        """
        f = intensities * telluric * self.area_atmosphere
        g = (stellar - intensities * self.area_planet) * telluric

        f = self.detector.apply_instrumental_broadening(f)
        g = self.detector.apply_instrumental_broadening(g)
        g = spectra - g

        # Each observation will have a different wavelength grid
        # in the planet restframe (since the planet moves quite fast)
        # therefore we use each wavelength point from each observation
        # individually, but we sort them by wavelength
        # so that the gradient, is still only concerned about the immediate
        # neighbours
        wave = []
        for time, w in zip(times, wavelength):
            rv = self.telescope_frame.to_frame(self.planet_frame, time)
            beta = (rv / c).to_value(1)
            w = np.copy(w) * np.sqrt((1 + beta) / (1 - beta))
            wave += [w]

        wave = np.concatenate(wave)
        f = f.ravel()
        g = g.ravel()
        idx = np.argsort(wave)
        wave = wave[idx]
        f, g = f[idx], g[idx]

        mask = np.isfinite(f) & np.isfinite(g)
        mask &= (f != 0) & (g != 0)
        wave = wave[mask]
        f, g = f[mask], g[mask]
        return wave, f, g

    def solve(
        self, times, wavelength, spectra, stellar, intensities, telluric, **kwargs
    ):
        pass
