# Loads TAPAS tellurics

import glob
import gzip
from os.path import dirname, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

import astropy.units as u

import astroplan
from .datasource import DataSource
from ..spectrum import Spectrum1D, SpectrumArray


def load_tellurics(files):
    telfil = glob.glob(files)  # reading the tellurics

    # Parse the header
    airmass, ang = np.zeros(np.size(telfil)), np.zeros(np.size(telfil))
    for i, ff in enumerate(telfil):
        with gzip.open(ff) as file:
            data = [file.readline().decode() for _ in range(23)]
            airmass[i] = np.float(data[15].strip()[9:])
            ang[i] = np.float(data[14].strip()[4:])

    # Parse the data
    tell = [None for _ in telfil]
    sort = np.argsort(airmass)
    airmass, ang = airmass[sort], ang[sort]
    for i in tqdm(range(len(airmass)), leave=False):
        # Pandas is faster at parsing tables than numpy
        buff = pd.read_table(
            telfil[sort[i]],
            skiprows=23,
            header=None,
            names=["wavelength", "transmittance"],
            skipinitialspace=True,
            sep=r"\s+",
        )
        tell[i] = buff.values
    # Combine all the data in the end
    tell = np.stack(tell)
    return tell, airmass, ang


class TapasTellurics(DataSource):
    def __init__(self, star, observatory, dataset="winter"):
        super().__init__()

        self.dataset = dataset

        self.star = star
        self.observatory = observatory
        # Define target parameterss
        coords = star.coordinates
        self.target = astroplan.FixedTarget(name=star.name, coord=coords)
        self.observer = astroplan.Observer(observatory)

        # Load data
        tapas_dir = join(dirname(__file__), "../../data/tapas/")
        tellw, self.airw, self.angw = load_tellurics(join(tapas_dir, "*winter*ipac.gz"))
        tells, self.airs, self.angs = load_tellurics(join(tapas_dir, "*summer*ipac.gz"))

        # Sort wavelength axis
        # We assume here that all files use the same wavelength axis
        wavew, waves = np.squeeze(tellw[0, :, 0]), np.squeeze(tells[0, :, 0])
        iiw, iis = np.argsort(wavew), np.argsort(waves)
        self.tellw, self.tells = tellw[:, iiw, :], tells[:, iis, :]

        # Create interpolator
        self.wavew, self.waves = (
            np.squeeze(self.tellw[0, :, 0]),
            np.squeeze(self.tells[0, :, 0]),
        )
        self.fluxw, self.fluxs = (
            np.squeeze(self.tellw[:, :, 1]),
            np.squeeze(self.tells[:, :, 1]),
        )
        self.tellwi = RegularGridInterpolator((self.airw, self.wavew), self.fluxw)
        self.tellsi = RegularGridInterpolator((self.airs, self.waves), self.fluxs)

    def calculate_airmass(self, time):
        """Determine the airmass for a given time
            
        Parameters
        ----------
        time : Time
            Time of the observation
        
        Returns
        -------
        airmass : float
            Airmass
        """
        altaz = self.observer.altaz(time, self.target)
        airmass = altaz.secz.value
        if np.any(airmass < 0):
            raise ValueError(
                "Nonsensical negative airmass was calculated, check your observation times"
            )
        return airmass

    def interpolate(self, airmass):
        if self.dataset == "winter":
            wave = self.wavew
            interpolator = self.tellwi
        elif self.dataset == "summer":
            wave = self.waves
            interpolator = self.tellsi
        else:
            raise ValueError

        if not hasattr(airmass, "__len__"):
            airmass = (airmass,)

        flux = np.empty((len(airmass), wave.size))
        for i, air in enumerate(airmass):
            flux[i] = interpolator((air, wave))

        spec = Spectrum1D(spectral_axis=wave << u.nm, flux=flux << u.one)
        return spec

    def get(self, wrange, time):
        airmass = self.calculate_airmass(time)
        spectra = self.interpolate(airmass)
        spectra = spectra.extract_region(wrange)

        spectra.description = "telluric transmission spectrum from a model"
        spectra.source = "TAPAS"
        spectra.datetime = time
        spectra.star = self.star
        spectra.observatory_location = self.observatory
        spectra.reference_frame = "telescope"

        return spectra
