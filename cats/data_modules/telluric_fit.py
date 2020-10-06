import astroplan
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from tqdm import tqdm

from scipy.optimize import least_squares, minimize
from scipy.sparse import diags
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

from .datasource import DataSource


class TelluricFit(DataSource):
    def __init__(self, star, observatory, degree=2, skip_resample=False):
        self.star = star
        self.observatory = observatory

        # Define target parameterss
        coords = star.coordinates
        self.target = astroplan.FixedTarget(name=star.name, coord=coords)
        self.observer = astroplan.Observer(observatory)

        self.degree = degree
        self.skip_resample = skip_resample

    def polyfit(self, x, y):
        coeff = np.polyfit(x, y, self.degree)
        # res = least_squares(lambda p: np.polyval(p, x) - y, x0=coeff, method="trf", loss="soft_l1")
        # coeff = res.x
        return coeff

    def spline_fit(self, coeff, s=1):
        p = np.arange(coeff.shape[0])
        for i in tqdm(range(coeff.shape[1])):
            weights = ~np.isnan(coeff[:, i])
            coeff[:, i][~weights] = 0
            coeff[:, i] = UnivariateSpline(p, coeff[:, i], s=s, w=weights)(p)
        return coeff

    def fit(self, spectra):
        times = spectra.datetime
        idx = np.argsort(times)[len(times) // 2]
        wavelength = spectra.wavelength[idx]

        # Get spectra
        if not self.skip_resample:
            resampled = spectra.resample(wavelength)
            flux = resampled.flux
        else:
            flux = spectra.flux

        # Calculate airmass
        airmass = self.calculate_airmass(times)

        # Fit data
        coeff = np.zeros((flux.shape[1], self.degree + 1))
        for i in tqdm(range(flux.shape[1])):
            coeff[i] = self.polyfit(airmass, flux[:, i].value)

        # coeff = self.spline_fit(coeff)

        return coeff

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

    def model(self, coeff, airmass):
        m = np.zeros((len(airmass), coeff.shape[0]))
        m += coeff[:, 0]
        for i in range(1, coeff.shape[1]):
            m *= airmass[:, None]
            m += coeff[:, i]

        return m
