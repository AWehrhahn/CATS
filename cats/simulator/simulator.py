"""
Simulate observations of a transiting planet
Adding various sources of noise

Input:
  - Stellar spectrum (with SME)
  - Stellar intensities (with SME)
  - Tellurics (with Molecfit?)
  - Planet spectrum (from PSG?)
  - 

"""
import numpy as np

from ExoOrbit import Orbit

from . import noise


class Simulator:
    def __init__(self, star, planet, stellar, intensities, telluric, planet_spectrum):
        # Orbit information
        self.planet = planet
        self.star = star
        self.orbit = Orbit(self.star, self.planet)
        # Input spectra
        self.telluric = telluric
        self.stellar = stellar
        self.intensities = intensities
        self.planet_spectrum = planet_spectrum
        # Noise parameters
        self.noise = [noise.WhiteNoise(0.01), noise.BadPixelNoise(0.02, 0.1)]

    @staticmethod
    def get_number_of_wavelengths_points_from_resolution(R, wmin, wmax):
        def gen(R, wmin, wmax):
            delta_wave = lambda w: w / R
            wave_local = wmin
            yield wave_local
            while wave_local < wmax:
                wave_local += delta_wave(wave_local)
                yield wave_local
            return

        generator = gen(R, wmin, wmax)
        ls = list(generator)
        return len(ls)

    def simulate_single(self, wave, time):
        """
        Simulate an observation using the current settings
        at the given datetime

        Parameters
        ----------
        wave : Quality
            wavelength grid of the observation to simulate
        phase : float
            phase of the observation, with 

        Returns
        -------
        obs : Spectrum1D
            simulated observation
        """

        # Generate intensities for the current mu
        mu = self.orbit.mu(time)
        i_core = self.intensities(mu, self.planet, "core")
        i_atm = self.intensities(mu, self.planet, "atmosphere")

        # Shift to telescope restframe
        frame = "telescope"
        planet_spectrum = self.planet_spectrum.shift(frame)
        stellar = self.stellar.shift(frame)
        telluric = self.telluric.shift(frame)
        i_core = i_core.shift(frame)
        i_atm = i_atm.shift(frame)

        # interpolate all onto the same wavelength grid
        method = "flux_conserved"
        planet_spectrum = planet_spectrum.resample(wave, method=method)
        stellar = stellar.resample(wave, method=method)
        telluric = telluric.resample(wave, method=method)
        i_core = i_core.resample(wave, method=method)
        i_atmo = i_atmo.resample(wave, method=method)

        # Observed spectrum
        area_planet = self.planet.area
        area_atm = np.pi * (self.planet.radius + self.planet.atm_height) ** 2

        obs = (stellar - i_core * area_planet + i_atmo * planet * area_atm) * telluric

        # Generate noise
        size = len(wave)
        noise = np.zeros(size)
        for n in self.noise:
            noise += n(size)

        # Apply instrumental broadening and noise
        obs *= 1 + noise

        return obs.data

    def simulate_series(self, nobs):
        """
        Simulate a series of observations
        
        Parameters
        ----------
        nobs : int
            number of equispaced observations

        Returns
        -------
        series : SpectrumList
            a list of observations
        """
        # Calculate phase
        period = self.parameters["period"].to("day").value
        t1 = self.orbit.first_contact() - period / 100
        t4 = self.orbit.fourth_contact() + period / 100
        time = np.linspace(t1, t4, self.n_obs)
        phase = self.orbit.phase_angle(time)

        # Create new wavelength grid
        wpoints = Simulator.get_number_of_wavelengths_points_from_resolution(
            self.R, self.wmin, self.wmax
        )
        wgrid = np.geomspace(self.wmin, self.wmax, wpoints)
        wgrid[0] = self.wmin
        wgrid[-1] = self.wmax

        # TODO: shift the wavelength grid a bit for each observation (as in real observations)

        s = self.simulate_singe(wgrid, time[0])
