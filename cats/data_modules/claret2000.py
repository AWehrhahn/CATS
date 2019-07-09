"""
A class to access limb darkening formulas from Claret 2000
"""

import logging
import os.path

from astropy.io import fits
import numpy as np
import pandas as pd

from .specific_intensity import data_intensities
from .dataset import dataset


class claret2000(data_intensities):
    """ access limb darkening formulas from Claret 2000 """

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
        return 1 - a[0] * (1 - mu**0.5) - \
            a[1] * (1 - mu) - a[2] * (1 - mu**1.5) - a[3] * (1 - mu**2)

    def load_intensities(self, **data):
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
        cls = claret2000
        par = data["parameters"]
        stellar = data["stellar_flux"]

        filename = self.configuration['file']
        folder = os.path.dirname(__file__)
        filename = os.path.join(folder, filename)

        hdulist = fits.open(filename)
        lddata = hdulist[1].data

        # Get stellar parameters from parameters and round to next best grid value
        T = cls.round_to(par['teff'], 250, limits=[3500, 4500])
        logg = cls.round_to(par['logg'], 0.5, limits=[0, 5])
        vmt = cls.round_to(par['star_vt'], 1, limits=[0, 8])
        met = cls.round_to(par['monh'], 0.1, limits=[-5, 0.5])
        if met == 0.4:
            met = 0.3

        if vmt == 3:
            vmt = 2
        if vmt in [5, 6]:
            vmt = 4
        if vmt == 7:
            vmt = 8

        logging.info('T_eff: %s, logg: %s, [M/H]: %s, micro_turbulence: %s', T, logg, met, vmt)

        # Select correct coefficients
        lddata = lddata[(lddata['Teff'] == T) & (lddata['logg'] == logg) & (
            lddata['VT'] == vmt) & (lddata['log_M_H_'] == met)]
        coeff = lddata['Coeff']
        names = ['U', 'B', 'V', 'R', 'I']
        wl = [3650, 4450, 5510, 6580, 8060]  # in Angström, from wikipedia
        values = {j: lddata[i] for i, j in zip(names, wl)}
        values = pd.DataFrame(data=values, index=coeff)

        # Interpolate to the wavelength grid of the observation
        # this is quite rough as there are not many wavelengths to choose from
        a1 = np.interp(stellar.wave, wl, values.loc['a1 '])
        a2 = np.interp(stellar.wave, wl, values.loc['a2 '])
        a3 = np.interp(stellar.wave, wl, values.loc['a3 '])
        a4 = np.interp(stellar.wave, wl, values.loc['a4 '])
        # Apply limb darkening to each point and set values of mu, and store the results

        a = [a1, a2, a3, a4]
        mus = self.configuration['star_intensities']
        if mus == "geom":
            mus = np.geomspace(1, 0.0001, num=20)
            mus[-1] = 0

        star_int = {i: stellar.data *
                    cls.limb_darkening_formula(i, a) for i in mus}
        star_int = pd.DataFrame(star_int)
        ds = dataset(stellar.wave, star_int)
        return ds
