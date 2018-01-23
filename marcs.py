"""
Load MARCS model stellar spectra
marcs.astro.uu.se
"""
import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.constants import c, pi
from scipy.integrate import simps, trapz
from scipy.interpolate import interp1d

from awlib.astro import air2vac, doppler_shift
from data_module_interface import data_module
from DataSources import Cache


class marcs(data_module):
    ###
    #  STELLAR FLUX
    ###

    @classmethod
    def apply_modifiers(cls, conf, par, wl, flux):
        if 'marcs_flux_mod' in conf.keys():
            flux *= float(conf['marcs_flux_mod'])
        if 'marcs_wl_mod' in conf.keys():
            wl *= float(conf['marcs_wl_mod'])
        return wl, flux

    @classmethod
    def load_stellar_flux_from_intensities(cls, config, par):
        """ load MARCS flux from specific intensities """
        wl, data = cls.load_data(config, par)

        imu = config['star_intensities']
        intensities = data

        #Integrate intensity * mu dmu 
        flux = simps(intensities * imu, imu) * (-2)
        wl, flux = cls.apply_modifiers(config, par, wl, flux)

        return wl, flux

    @classmethod
    def load_stellar_flux(cls, conf, par, fname=None, apply_air2vac=True):
        """ Load MARCS flux directly from flx file """
        if fname is None:
            fname = join(conf['input_dir'], conf['marcs_dir'],
                         conf['marcs_file_flux'])
        df = pd.read_table(fname, delim_whitespace=True, names=[
            'wl', 'rel_flux', 'abs_flux', 'rel_flux_conv', 'abs_flux_conv'], skiprows=1)

        wl = df['wl'].values
        flux = df['abs_flux'].values
        wl, flux = cls.apply_modifiers(conf, par, wl, flux)

        if apply_air2vac:
            wl = air2vac(wl)

        flux = doppler_shift(wl, flux, par['radial_velocity'])
        #flux /= (2 * np.pi) #change to physical flux #TODO

        return wl, flux

    ###
    #  LIMB DARKENING
    ###

    @classmethod
    def linear(cls, new, old, values):
        # TODO use spline interpolation?
        return np.interp(new, old, values)

    @classmethod
    def spline(cls, new, old, values):
        return interp1d(old, values, kind='quadratic')(new)

    @classmethod
    def read(cls, fname, imu, interpolate='linear'):
        """
        reads one limb darkening file and interpolates the values to the given mu values
        fname: filename
        imu: mu values to interpolate to
        interpolate: interpolation function to use
        """

        if interpolate in ['l', 'linear']:
            interpolate = cls.linear
        elif interpolate in ['s', 'spline']:
            interpolate = cls.spline
        else:
            raise AttributeError('Interpolation method not recognised')

        nwl = 150000  # TODO upper limit of entries
        result = np.zeros((nwl, len(imu) + 1))

        f = open(fname, 'rb')
        # The first three lines are header
        f.seek(48, os.SEEK_SET)

        # format
        # double8, int4, (real4, real4)
        # xl(j),mmu,(xmu(nlx),y1(nlx),nlx=1,mmu)

        # xl: wavelength in Angstrom
        # mmu: number of mu points for this wavelength
        # xmu: the mu points
        # y1: the spectrum at these mu values for this wavelength

        for i in range(nwl):
            # data type of first half
            dt = np.dtype([('xl', '>f8'), ('mmu', '>i4')])
            content = np.fromfile(f, dtype=dt, count=1)
            if len(content) == 0:
                # end of file
                break

            wl, nmu = content[0]

            # data type of second half
            at = np.dtype(('>f4', (nmu, 2)))
            content = np.fromfile(f, dtype=at, count=1)[0]

            xmu, y1 = content.swapaxes(0, 1)
            # inverse order
            xmu = -xmu[::-1]  # flip sign to positive
            y1 = y1[::-1]

            y2 = interpolate(imu, xmu, y1)

            result[i] = [wl, *y2]

            # skip to end of line
            f.read(8)

        result = result[:i]
        return result

    @classmethod
    def read_all(cls, fname, imu, ld_format, interpolate='linear'):
        """
        read several files in a series
        fname: filename unformated
        imu: range of mu values
        frim: first value to use in fname.format
        to: last value
        interpolate: interpolation method
        """
        result = [cls.read(fname.format(i), imu, interpolate=interpolate)[100:-100]
                  for i in ld_format]

        result = np.concatenate(result)[:-200]
        return result

    @classmethod
    def load_data(cls, config, par):
        # filename
        flux_file = join(config['input_dir'],
                         config['marcs_dir'], config['marcs_file_ld'])
        imu = config['star_intensities']
        ld_format = config['marcs_ld_format']
        interpolation = config['marcs_ld_interpolation_method']
        cache = Cache.Cache(config['path_cache'], par['name_star'],
                            par['name_planet'], flux_file, ld_format, imu, interpolation)

        try:
            data = cache.load()
        except OSError:
            data = None

        if data is None:
            data = cls.read_all(flux_file, imu, ld_format, interpolation)
            cache.save(data)

        # Air to Vacuum conversion
        wl = air2vac(data[:, 0])

        # Doppler shift
        v = par['radial_velocity']
        c_loc = c * 1e-3
        shift = (1 + v / c_loc) * wl
        for i in range(1, data.shape[1]):
            data[:, i] = np.interp(wl, shift, data[:, i])

        return data[:, 0], data[:, 1:]

    @classmethod
    def load_specific_intensities(cls, config, par, *args, **kwargs):
        imu = config['star_intensities']
        wl, result = cls.load_data(config, par)
        wl, result = cls.apply_modifiers(config, par, wl, result)
        df = pd.DataFrame(result, columns=imu)
        return wl, df

    @classmethod
    def load_limb_darkening(cls, config, par):
        #TODO do I need a factor mu somewhere?
        # to convert from specific intensity to flux I had to integrate with mu
        wl_i, intensities = cls.load_intensities(config, par)
        wl_f, flux = cls.load_flux_directly(config, par)

        flux = np.interp(wl_i, wl_f, flux)
        intensities = intensities.apply(lambda s: s / flux)
        return wl_i, intensities

    ###
    # Solar Model
    ###
    @classmethod
    def load_solar(cls, conf, par, calib_dir):
        s_fname = join(calib_dir, 'sun.flx')
        s_wave, s_flux = cls.load_stellar_flux(
            conf, par, s_fname, apply_air2vac=False)
        return s_wave, s_flux / (2*np.pi)
