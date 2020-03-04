from os.path import dirname, join

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from specutils.spectra import SpectralRegion

from ..spectrum import Spectrum1D

class Detector:
    def __init__(self):
        super().__init__()
        self.regions = None
        self.blaze = None
        self.noise = []
        #:int: Pixels per order
        self.pixels = 0

class Crires(Detector):
    def __init__(self, setting="H/1/4", detector=1):
        super().__init__()
        self.setting = setting
        self.detector = detector
        self.pixels = 2048
        self.regions = self.__class__.load_spectral_regions(setting, detector)
        self.blaze = self.__class__.load_blaze_function(setting, detector)

    @staticmethod
    def load_spectral_regions(setting, detector):
        # Spectral regions
        # from https://www.astro.uu.se/crireswiki/Instrument?action=AttachFile&do=view&target=crmcfgWLEN_20200223_extracted.csv
        fname = join(dirname(__file__), "crires_wlen_extracted.csv")
        data= pd.read_csv(fname, skiprows=[1])

        idx = data["setting"] == setting
        regions = []
        for order in range(1, 9):
            wmin = data[f"O{order} BEG DET{detector}"][idx].array[0] * u.nm
            wmax = data[f"O{order} END DET{detector}"][idx].array[0] * u.nm
            regions += [SpectralRegion(wmin, wmax)]

        regions = np.sum(regions)
        return regions

    @staticmethod
    def load_blaze_function(setting, detector):
        # TODO: have a datafile, that has the different blaze functions
        # for the different settings
        fname = join(dirname(__file__), "crires_blaze.txt")
        data = np.genfromtxt(fname)
        norders = len(data)
        blaze = np.zeros((norders, 2048)) << u.one

        x = np.arange(2048) + 1
        for order in range(8):
            b = np.polyval(data[order], x)
            blaze[order] = b << u.one

        return blaze

    @staticmethod
    def load_noise_parameters(setting, detector):
        # TODO add noise profiles
        noise = []
        return noise
