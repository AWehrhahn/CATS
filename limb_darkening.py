"""
A class to access limb darkening formulas from Claret 2000
"""
import astropy.io.fits as fits
import numpy as np
import pandas as pd

from data_module_interface import data_module
from dataset import dataset


class limb_darkening(data_module):
    """ access limb darkening formulas from Claret 2000 """

    @classmethod
    def round_to(cls, n, precision, limits=None):
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

        """
        Round a value to the next closest value with precision p inside the limits
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

    @classmethod
    def load_specific_intensities(cls, config, par, stellar):
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
        cls.log(2, 'Limb darkening formula by Claret 2000')
        file_limb_darkening = config['ld_file']
        hdulist = fits.open(file_limb_darkening)
        lddata = hdulist[1].data

        # Get stellar parameters from parameters and round to next best grid value
        T = cls.round_to(par['star_temp'], 250, limits=[3500, 4500])
        logg = cls.round_to(par['star_logg'], 0.5, limits=[0, 5])
        vmt = cls.round_to(par['star_vt'], 1, limits=[0, 8])
        met = cls.round_to(par['star_metallicity'], 0.1, limits=[-5, 0.5])
        if met == 0.4:
            met = 0.3

        if vmt == 3:
            vmt = 2
        if vmt in [5, 6]:
            vmt = 4
        if vmt == 7:
            vmt = 8

        cls.log(2, 'T_eff: %s, logg: %s, met: %s, micro_turbulence: %s' % (T, logg, met, vmt))

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
        a1 = np.interp(stellar.wl, wl, values.loc['a1 '])
        a2 = np.interp(stellar.wl, wl, values.loc['a2 '])
        a3 = np.interp(stellar.wl, wl, values.loc['a3 '])
        a4 = np.interp(stellar.wl, wl, values.loc['a4 '])
        # Apply limb darkening to each point and set values of mu, and store the results

        a = [a1, a2, a3, a4]
        mus = config['star_intensities']
        star_int = {i: stellar.flux[0] *
                    cls.limb_darkening_formula(i, a) for i in mus}
        star_int = pd.DataFrame(star_int)
        ds = dataset(stellar.wl, star_int)
        return ds
