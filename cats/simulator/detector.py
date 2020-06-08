from os.path import dirname, join

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from specutils.spectra import SpectralRegion

from astropy.coordinates import EarthLocation

from .noise import PoisonNoise, WhiteNoise


class Detector:
    def __init__(self):
        super().__init__()
        self.regions = None
        self.blaze = None
        self.noise = []
        #:int: Pixels per order
        self.pixels = 0
        self.pixel_size = 0 * u.nm
        self.collection_area = 0 * u.m ** 2
        self.integration_time = 0 * u.second
        self.gain = 1  # Number of electrons / photon
        self.readnoise = 0
        self.efficiency = 1


class Crires(Detector):
    def __init__(self, setting="H/1/4", detector=1, orders=None):
        super().__init__()
        self.setting = setting
        self.detector = detector
        self.orders = orders
        self.pixels = 2048
        self.pixel_size = 18 * u.um
        self.collection_area = (8 * u.m) ** 2
        self.integration_time = 5 * u.min
        self.bad_pixel_ratio = 4e5 / (2048 ** 2)
        self.spectral_broadening = 1
        self.observatory = EarthLocation.of_site("Cerro Paranal")

        # TODO: gain / readnoise for each detector / wavelength range
        self.gain = [2.15, 2.19, 2.00]
        self.readnoise = [11, 12, 12]
        self.efficiency = 0.5  # About 50 percent makes it though

        self.regions, self.norders = self.__class__.load_spectral_regions(
            setting, detector, orders
        )
        self.blaze = self.__class__.load_blaze_function(setting, detector, orders)

        assert (
            len(self.regions) == len(self.blaze) == sum(self.norders)
        ), "Incompatible sizes, something went wrong"

        # Expand detector values to the wavelength regions
        self.gain = np.repeat(self.gain, self.norders)
        self.readnoise = np.repeat(self.readnoise, self.norders)

    def __str__(self):
        return "CRIRES"

    @staticmethod
    def load_spectral_regions(setting: str, detector: list, orders: list):
        # Spectral regions
        # from https://www.astro.uu.se/crireswiki/Instrument?action=AttachFile&do=view&target=crmcfgWLEN_20200223_extracted.csv
        fname = join(dirname(__file__), "crires_wlen_extracted.csv")
        data = pd.read_csv(fname, skiprows=[1])

        detector = np.atleast_1d(detector)
        norders = [0 for _ in detector]

        if orders is None:
            orders = range(1, 11)

        idx = data["setting"] == setting
        regions = []
        for order in orders:
            for i, det in enumerate(detector):
                if data[f"O{order} Central Wavelength"][idx].array[0] != -1:
                    wmin = data[f"O{order} BEG DET{det}"][idx].array[0] * u.nm
                    wmax = data[f"O{order} END DET{det}"][idx].array[0] * u.nm
                    regions += [SpectralRegion(wmin, wmax)]
                    norders[i] += 1

        # regions = np.sum(regions)
        return regions, norders

    @staticmethod
    def parse_blaze_file(fname):
        with open(fname, "r") as file:
            lines = file.readlines()

        counter = 0
        data = {}
        while counter < len(lines) - 1:
            setting = lines[counter][12:].strip()
            counter += 1
            data[setting] = {}
            # detector = lines[counter]
            counter += 1
            for det in [1, 2, 3]:
                coeff = []
                while counter < len(lines):
                    line = lines[counter]
                    counter += 1
                    if line[:3] in ["DET", "STD"]:
                        break
                    line = [float(s) for s in line.split()]
                    coeff += [line]

                coeff = np.array(coeff)
                data[setting][det] = coeff
            counter -= 1

        return data

    @staticmethod
    def load_blaze_function(setting: str, detector: list, orders: list):
        # TODO: have a datafile, that has the different blaze functions
        # for the different settings
        setting = setting.replace("/", "_")
        fname = join(dirname(__file__), "crires_blaze.txt")
        detector = np.atleast_1d(detector)
        data = Crires.parse_blaze_file(fname)

        x = np.arange(2048) + 1

        if orders is None:
            norders = sum([len(data[setting][det]) for det in detector])
        else:
            norders = len(detector) * len(orders)

        blaze = np.zeros((norders, 2048)) << u.one
        counter = 0
        for det in detector:
            if orders is None:
                _orders = range(len(data[setting][det]))
            else:
                _orders = range(len(orders))
            for order in _orders:
                b = np.polyval(data[setting][det][order], x)
                blaze[counter] = b << u.one
                counter += 1

        blaze /= np.max(blaze)

        return blaze

    def load_noise_parameters(self):
        # TODO: Add noise profiles
        # TODO: Noise depends on the detector, so have seperate Noises for each segment?
        # for maximum flexibility

        noise = []
        # We spectrum is the sum over several pixels
        # so the individual noise is reduced by a factor of scaling
        # TODO: the exact factor depends on the slitfunction of the spectrum
        # For a gaussian with width 10: 0.28
        # For a gaussuan with width 20: 0.12
        scaling = 0.28
        # Readnoise is just Gaussian
        readnoise = self.readnoise * scaling
        noise += [WhiteNoise(readnoise)]

        # Shotnoise depends on the measured spectrum
        noise += [PoisonNoise(scaling)]

        # Bad Pixel Noise depends on the number of bad pixels in the spectrum
        # We assume a random distribution without bias over the whole detector

        # Dark current noise
        # Its just Gaussian?

        return noise

    def apply_instrumental_broadening(self, spec):
        # TODO what is the psf?
        sigma = self.spectral_broadening
        if hasattr(spec, "flux") and not hasattr(spec, "__len__"):
            flux = spec.flux.decompose()
            spec._unit = u.one
            spec._data[:] = gaussian_filter1d(flux, sigma)
        elif hasattr(spec, "flux") and hasattr(spec, "__len__"):
            for s in spec:
                flux = s.flux.decompose()
                s._unit = u.one
                s._data[:] = gaussian_filter1d(flux, sigma)
        else:
            spec = gaussian_filter1d(spec, sigma)

        return spec
