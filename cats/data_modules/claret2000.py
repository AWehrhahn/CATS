"""
A class to access limb darkening formulas from Claret 2000
"""

import logging
import os.path

from astropy.io import fits
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


from ..spectrum import Spectrum1D
from .datasource import StellarIntensities

logger = logging.getLogger(__name__)


class Claret2000(StellarIntensities):
    """ access limb darkening formulas from Claret 2000 """

    def __init__(self, star, planet, stellar):
        super().__init__(star, planet)
        self.load_data()

    @staticmethod
    def interpolate_intensity(mu, i):
        """ Interpolate the stellar intensity for given limb distance mu

        use linear interpolation, because it is much faster

        Parameters:
        ----------
        mu : {float, np.ndarray}
            cos(limb distance), i.e. 1 is the center of the star, 0 is the outer edge
        i : {pd.DataFrame}
            specific intensities
        Returns
        -------
        intensity : np.ndarray
            interpolated intensity
        """

        values = i.values.swapaxes(0, 1)
        keys = np.asarray(i.keys())
        flux = interp1d(
            keys,
            values,
            kind="linear",
            axis=0,
            bounds_error=False,
            fill_value=(values[0], values[-1]),
        )(mu)
        flux[mu < 0, :] = 0
        return flux

    @staticmethod
    def round_to(n, precision, limits=None):
        """ Round to the closest value within the given precison or the next limit

        Parameters:
        ----------
        n : {float}
            value to round
        precision : {float}
            precision of the rounding, e.g. 0.5
        limits : {tuple(min, max), None}, optional
            Limits the results to min, max, or no limits if None (the default is None)

        Returns
        -------
        rounded : float
            rounded value
        """
        correction = 0.5 if n >= 0 else -0.5
        value = int(n / precision + correction) * precision
        if limits is None:
            return value

        if value >= limits[0] and value <= limits[1]:
            return value

        if value < limits[0]:
            return limits[0]

        if value > limits[1]:
            return limits[1]

    @staticmethod
    def limb_darkening_formula(mu, a):
        """ Limb darkening formula

        from Claret 2000

        Parameters:
        ----------
        mu : {float}
            limb distance
        a : {(a1, a2, a3, a4) float}
            limb darkening parameters
        Returns
        -------
        factor : float
            limb darkening factor
        """
        return (
            1
            - a[0] * (1 - mu ** 0.5)
            - a[1] * (1 - mu)
            - a[2] * (1 - mu ** 1.5)
            - a[3] * (1 - mu ** 2)
        )

    def get(self, wave, mu, mode="core"):
        """ Interpolate on the grid """
        star_int = self.load_intensities(self.stellar)

        spec = Claret2000.interpolate_intensity(mu, star_int)

        # TODO: citation
        ds = Spectrum1D(
            flux=spec,
            spectral_axis=wave,
            planet=self.planet,
            star=self.star,
            source="claret2000",
            description="Stellar intensities from limb darkening formula",
            citation="http://vizier.cfa.harvard.edu/viz-bin/VizieR?-source=J/A+A/363/1081",
        )
        return ds

    def load_data(self):
        filename = self.config["file"]
        folder = os.path.dirname(__file__)
        filename = os.path.join(folder, filename)

        hdulist = fits.open(filename)
        lddata = hdulist[1].data

    def load_intensities(self, stellar):
        """ Use limb darkening formula to estimate specific intensities

        The limb distances to evaluate are set in config
        formula from Claret 2000
        paramters from
        http://vizier.cfa.harvard.edu/viz-bin/VizieR?-source=J/A+A/363/1081

        Parameters:
        ----------
        config : {dict}
            configuration settings
        par : {dict}
            stellar and planetary parameters
        stellar : {dataset}
            stellar flux
        Returns
        -------
        spec_intensities : dataset
            specific intensities for several limb distances mu
        """
        cls = Claret2000

        # Get stellar parameters from parameters and round to next best grid value
        T = cls.round_to(self.star.teff.to("K").value, 250, limits=[3500, 4500])
        logg = cls.round_to(self.star.logg.value, 0.5, limits=[0, 5])
        vmt = cls.round_to(self.star.vturb.to("km/s").value, 1, limits=[0, 8])
        met = cls.round_to(self.star.monh.value, 0.1, limits=[-5, 0.5])
        if met == 0.4:
            met = 0.3
        if vmt == 3:
            vmt = 2
        if vmt in [5, 6]:
            vmt = 4
        if vmt == 7:
            vmt = 8

        logging.info(
            "T_eff: %s, logg: %s, [M/H]: %s, micro_turbulence: %s", T, logg, met, vmt
        )

        # Select correct coefficients
        lddata = self.lddata[
            (self.lddata["Teff"] == T)
            & (self.lddata["logg"] == logg)
            & (self.lddata["VT"] == vmt)
            & (self.lddata["log_M_H_"] == met)
        ]
        coeff = lddata["Coeff"]
        names = ["U", "B", "V", "R", "I"]
        wl = [3650, 4450, 5510, 6580, 8060]  # in Angström, from wikipedia
        values = {j: lddata[i] for i, j in zip(names, wl)}
        values = pd.DataFrame(data=values, index=coeff)

        # Interpolate to the wavelength grid of the observation
        # this is quite rough as there are not many wavelengths to choose from
        wave = stellar.wavelength
        a1 = np.interp(wave, wl, values.loc["a1 "])
        a2 = np.interp(wave, wl, values.loc["a2 "])
        a3 = np.interp(wave, wl, values.loc["a3 "])
        a4 = np.interp(wave, wl, values.loc["a4 "])
        # Apply limb darkening to each point and set values of mu, and store the results

        a = [a1, a2, a3, a4]
        mus = self.config["star_intensities"]
        if mus == "geom":
            mus = np.geomspace(1, 0.0001, num=20)
            mus[-1] = 0

        star_int = {i: stellar.flux * cls.limb_darkening_formula(i, a) for i in mus}
        star_int = pd.DataFrame(star_int)

        return star_int
