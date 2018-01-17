import astropy.io.fits as fits
import pandas as pd
import numpy as np

from data_module_interface import data_module

class limb_darkening(data_module):

    @classmethod
    def round_to(cls, n, precision, limits=None):
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
    def limb_darkening(mu, a):
        """ Limb darkening fomula by Claret 2000 """
        return 1 - a[0] * (1 - mu**0.5) - \
            a[1] * (1 - mu) - a[2] * (1 - mu**1.5) - a[3] * (1 - mu**2)


    @classmethod
    def load_specific_intensities(cls, config, par, wl_flux, flux):
        # Limb darkening
        # I(mu)/I(1) = 1 - a1 * (1-mu**1/2) -a2 * (1-mu) - a3 * (1-mu**3/2)-a4*(1-mu**2)
        # from Claret 2000, http://vizier.cfa.harvard.edu/viz-bin/VizieR?-source=J/A+A/363/1081
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
        a1 = np.interp(wl_flux, wl, values.loc['a1 '])
        a2 = np.interp(wl_flux, wl, values.loc['a2 '])
        a3 = np.interp(wl_flux, wl, values.loc['a3 '])
        a4 = np.interp(wl_flux, wl, values.loc['a4 '])

        # Apply limb darkening to each point and set values of mu, and store the results

        a = [a1, a2, a3, a4]
        mus = config['star_intensities']
        star_int = {i: flux * limb_darkening(i, a) for i in mus}

        star_int = pd.DataFrame.from_dict(star_int)
        return wl_flux, star_int