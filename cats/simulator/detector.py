from os.path import dirname, join

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from specutils.spectra import SpectralRegion

from ..spectrum import Spectrum1D
from .noise import WhiteNoise

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
        self.gain = 1 # Number of electrons / photon
        self.readnoise = 0
        self.efficiency = 1

class Crires(Detector):
    def __init__(self, setting="H/1/4", detector=1):
        super().__init__()
        self.setting = setting
        self.detector = detector
        self.pixels = 2048
        self.pixel_size = 18 * u.um
        self.collection_area = (8 * u.m) ** 2
        self.integration_time = 5 * u.min
        
        self.gain = 2
        self.readnoise = 0
        self.efficiency = 0.5 # About 50 percent makes it though

        self.regions = self.__class__.load_spectral_regions(setting, detector)
        self.blaze = self.__class__.load_blaze_function(setting, detector)

    @staticmethod
    def load_spectral_regions(setting, detector):
        # Spectral regions
        # from https://www.astro.uu.se/crireswiki/Instrument?action=AttachFile&do=view&target=crmcfgWLEN_20200223_extracted.csv
        fname = join(dirname(__file__), "crires_wlen_extracted.csv")
        data= pd.read_csv(fname, skiprows=[1])

        detector = np.atleast_1d(detector)

        idx = data["setting"] == setting
        regions = []
        for det in detector:
            for order in range(1, 11):
                if data[f"O{order} Central Wavelength"][idx].array[0] != -1:
                    wmin = data[f"O{order} BEG DET{det}"][idx].array[0] * u.nm
                    wmax = data[f"O{order} END DET{det}"][idx].array[0] * u.nm
                    regions += [SpectralRegion(wmin, wmax)]

        regions = np.sum(regions)
        return regions

    @staticmethod
    def parse_blaze_file(fname):
        with open(fname, "r") as file:
            lines = file.readlines()
        
        counter = 0
        data = {}
        while counter < len(lines)-1:
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
    def load_blaze_function(setting, detector):
        # TODO: have a datafile, that has the different blaze functions
        # for the different settings
        setting = setting.replace("/", "_")
        fname = join(dirname(__file__), "crires_blaze.txt")
        detector = np.atleast_1d(detector)
        data = Crires.parse_blaze_file(fname)

        x = np.arange(2048) + 1

        norders = sum([len(data[setting][det]) for det in detector])
        blaze = np.zeros((norders, 2048)) << u.one
        counter = 0
        for det in detector:
            for order in range(len(data[setting][det])):
                b = np.polyval(data[setting][det][order], x)
                blaze[counter] = b << u.one
                counter += 1

        # blaze /= np.max(blaze)

        return blaze

    @staticmethod
    def load_noise_parameters(setting, detector):
        # TODO: Add noise profiles
        # TODO: Noise depends on the detector, so have seperate Noises for each segment?
        # for maximum flexibility

        noise = []
        # Readnoise is just Gaussian
        readnoise = 0.01
        noise += [WhiteNoise(readnoise)]

        # Shotnoise depends on the measured spectrum

        # Bad Pixel Noise depends on the number of bad pixels in the spectrum
        # We assume a random distribution without bias over the whole detector

        # Dark current noise
        # Its just Gaussian?

        return noise
