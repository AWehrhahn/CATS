from os.path import dirname, join

import astroplan
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from scipy.interpolate import interp1d
from specutils.manipulation import extract_region
from specutils.spectra import SpectralRegion

from ..reference_frame import TelescopeFrame
from ..spectrum import Spectrum1D, SpectrumList
from .datasource import DataSource, StellarIntensities


class TelluricModel(DataSource):
    """
    Telluric Data based on a few fixed Spectra, with given airmass.
    The data will be interpolated to the airmass at the time of the
    observation.
    """

    def __init__(self, star, observatory):
        super().__init__()
        self.star = star
        self.observatory = observatory

        # Define target parameterss
        coords = star.coordinates
        self.target = astroplan.FixedTarget(name=star.name, coord=coords)
        self.observer = astroplan.Observer(observatory)

        # Load telluric data
        self.data_directory = self.config["data_directory"]
        self.data_directory = self.data_directory.format(fileDir=dirname(__file__))

        self.data_files = self.config["data_files"]

        self.points = None
        self.spectra = None
        self.load_data_all()

    def load_data_one(self, fname):
        """Load one telluric spectrum from disk

        Parameters
        ----------
        fname : str
            filename of that telluric spectrum

        Returns
        -------
        spec : Spectrum1D
            Telluric spectrum
        """
        hdu = fits.open(fname)
        data = hdu[1].data

        data_wave = data["wave"] << u.AA
        data_flux = data["flux"] << u.one
        spec = Spectrum1D(flux=data_flux, spectral_axis=data_wave)
        return spec

    def load_data_all(self):
        """Load the telluric data from disk"""
        nspectra = len(self.data_files)

        # This assumes all the data files are correctly formated
        xp = np.zeros(nspectra)
        yp = [None for _ in self.data_files]
        for i, value in enumerate(self.data_files):
            xp[i] = value["airmass"]
            yp[i] = self.load_data_one(join(self.data_directory, value["filename"]))

        self.wavelength = yp[0].wavelength
        self.points = xp
        self.spectra = np.array([np.asarray(y.flux) for y in yp])

    def interpolate_spectra(self, airmass):
        """Interpolate the stored telluric spectra to the desired airmass

        Parameters
        ----------
        airmass : float
            Airmass to interpolate

        Returns
        -------
        spec : Spectrum1D
            Telluric spectrum at the desired airmass
        """
        flux = interp1d(
            self.points, self.spectra.T, "linear", fill_value="extrapolate"
        )(airmass)
        flux = np.clip(flux, 0, 1)

        wave = self.wavelength
        flux = flux << u.one

        spec = Spectrum1D(flux=flux, spectral_axis=wave)
        return spec

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

    def get(self, wrange, time):
        """
        Get the telluric spectrum for given wavelength regions at certain time.
        The exact spectrum depends on the airmass that is present at that time.

        Parameters
        ----------
        wrange : SpectralRegion
            Wavelength range(s) that we want the telluric spectrum for. In the barycentric restframe.
        time : Time
            Time of the observation

        Returns
        -------
        SpectrumList
            Telluric spectra for the given time. The list contains one Spectrum1D for each wavelength subregion
        """
        airmass = self.calculate_airmass(time)
        spec = self.interpolate_spectra(airmass)

        wave, flux = [], []
        for i in range(len(wrange)):
            subrange = wrange[i]
            s = spec.extract_region(subrange)
            wave += [s.wavelength]
            flux += [s.flux]

        spectra = SpectrumList(
            flux=flux,
            spectral_axis=wave,
            description="telluric transmission spectrum from a model",
            source="Evangelos/CRIRES+ wiki",
            datetime=time,
            star=self.star,
            observatory_location=self.observatory,
            reference_frame="telescope",
        )

        return spectra
