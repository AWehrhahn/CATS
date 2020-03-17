"""
Simulate observations of a transiting planet
Adding various sources of noise

Input:
  - Stellar spectrum (with SME)
  - Stellar intensities (with SME)
  - Tellurics
  - Planet spectrum (from PSG)
  - Noise sources

"""
import astropy.constants as const
import astropy.units as u
import numpy as np
from astropy.time import Time
from tqdm import tqdm

import matplotlib.pyplot as plt

from exoorbit import Orbit

from ..reference_frame import TelescopeFrame
from . import noise as NoiseModel


class Simulator:
    def __init__(
        self,
        detector,
        star,
        planet,
        stellar,
        intensities,
        telluric,
        planet_spectrum,
        mu=np.geomspace(0.01, 1, 7),
        R=100_000,
        noise=None,
    ):
        self.detector = detector
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
        if noise is None:
            self.noise = [
                NoiseModel.WhiteNoise(0.01),
                NoiseModel.BadPixelNoise(0.02, 0.1),
            ]
        else:
            self.noise = noise

        # Spectral resolution
        self.R = R

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

    def simulate_single(self, wrange, time):
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
        telluric = self.telluric.get(wrange, time)
        i_core = self.intensities.get(wrange, time, "core")
        i_atmo = self.intensities.get(wrange, time, "atmosphere")
        planet_spectrum = self.planet_spectrum.get(wrange, time)
        stellar = self.stellar.get(wrange, time)

        # Check that there is a planet spectrum at all!!
        # for i in range(len(planet_spectrum)):
        #     plt.plot(planet_spectrum[i].wavelength, planet_spectrum[i].flux)
        # plt.show()

        wave = self.create_wavelength(wrange)
        blaze = self.detector.blaze

        # Shift to telescope restframe
        observatory_location = self.detector.observatory
        sky_location = self.star.coordinates
        frame = TelescopeFrame(observatory_location, sky_location)
        planet_spectrum = planet_spectrum.shift(frame)
        stellar = stellar.shift(frame)
        telluric = telluric.shift(frame)
        i_core = i_core.shift(frame)
        i_atmo = i_atmo.shift(frame)

        # interpolate all onto the same wavelength grid
        method = "linear"
        planet_spectrum = planet_spectrum.resample(wave, method=method)
        stellar = stellar.resample(wave, method=method)
        telluric = telluric.resample(wave, method=method)
        i_core = i_core.resample(wave, method=method)
        i_atmo = i_atmo.resample(wave, method=method)

        # Observed spectrum
        area_planet = self.planet.area / self.star.area
        area_atm = np.pi * (self.planet.radius + self.planet.atm_scale_height) ** 2
        area_atm /= self.star.area

        obs = (
            stellar - i_core * area_planet + (i_atmo * planet_spectrum) * area_atm
        ) * telluric

        # Distance modulus
        obs *= (self.star.radius / self.star.distance).decompose() ** 2

        # Convert units to number of photons
        wave_bin = [np.gradient(wave) for wave in obs.wavelength]
        obs *= wave_bin
        obs *= self.detector.collection_area * self.detector.integration_time

        photon_energy = [wave / (const.h * const.c) for wave in obs.wavelength]
        obs *= photon_energy

        # Detector efficiency and gain to determine ADUs
        obs *= self.detector.efficiency / self.detector.gain

        # TODO: Why this factor?
        # The height of the order? No, the total is the sum of all values
        # but then the expected value of the spectrum is larger than i thought
        # obs *= 1 / self.detector.order_height

        # Apply blaze function
        # TODO: blaze is given as the flat field measurement, so what does that mean?
        obs *= blaze

        # Instrumental broadening
        obs = self.detector.apply_instrumental_broadening(obs)

        # Various Noise sources
        size = wave.shape
        data = obs.flux
        noise = np.zeros(size)
        for source in self.noise:
            noise += source(size, data)
        obs += noise

        for spec in obs:
            spec.meta["star"] = self.star
            spec.meta["planet"] = self.planet
            spec.meta["datetime"] = time
            # spec.reference_frame = frame

        return obs

    def simulate_series(self, wrange, time, nobs):
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
        self.orbit.planet.time_of_transit = time

        duration = self.planet.transit_duration.to_value("day")
        t1 = self.orbit.first_contact().mjd - duration / 2
        t4 = self.orbit.fourth_contact().mjd + duration / 2
        time = Time(np.linspace(t1, t4, nobs), format="mjd")

        obstime = (time[-1] - time[0]) / nobs
        self.detector.integration_time = obstime.jd * u.day

        # do the calculations only once
        self.intensities.prepare(wrange, time)

        # TODO: shift the wavelength grid a bit for each observation (as in real observations) ??
        # TODO: optimize sme calculations (i.e. do all mu values at the same time)
        spectra = []
        for t in tqdm(time, desc="Observation"):
            spectra += [self.simulate_single(wrange, t)]
        return spectra

    def create_wavelength(self, wrange):
        # Create new wavelength grid
        norders = len(wrange.subregions)
        npixels = self.detector.pixels
        wave = np.zeros((norders, npixels)) << u.AA

        for i, (wmin, wmax) in enumerate(wrange.subregions):
            wmin = wmin.to_value(u.AA)
            wmax = wmax.to_value(u.AA)

            wgrid = np.geomspace(wmin, wmax, npixels)
            wgrid[[0, -1]] = wmin, wmax

            wave[i] = wgrid << u.AA

        return wave
