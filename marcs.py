"""
Load MARCS model stellar spectra
marcs.astro.uu.se
"""
import os
from os.path import join

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import trapz, simps
from scipy.interpolate import interp1d
from scipy.constants import c, pi

from DataSources import Cache
from awlib.astro import air2vac

###
#  STELLAR FLUX
###


def scaling_factor(par):
    dist = 1 / (par['parallax'] * 1e-3) * 3.086e16  # in m
    h = 10**2  # m**2 #Telescope size
    return h / dist**2


def load_flux(config, par):
    """ load MARCS flux from specific intensities """
    wl, data = load_data(config, par)

    imu = config['star_intensities']
    intensities = data

    flux = simps(intensities * imu, imu) * (-2) 

    """
    r = np.sqrt(1 - imu**2)
    i_tmp = [intensities[:, 0] * np.pi * r[0]**2]
    for j in range(1, len(imu)):
        i_tmp.append(intensities[:, j] * np.pi * (r[j]**2 - r[j - 1]**2))

    flux = np.sum(i_tmp, 0) / np.pi
    """
    if 'marcs_flux_mod' in config.keys():
        flux *= float(config['marcs_flux_mod'])
    if 'marcs_wl_mod' in config.keys():
        wl *= config['marcs_wl_mod']

    #flux += -0.000102
    return wl, flux


def load_flux_directly(conf, par):
    """ Load MARCS flux directly from flx file """
    fname = join(conf['input_dir'], conf['marcs_dir'], conf['marcs_file_flux'])
    df = pd.read_table(fname, delim_whitespace=True, names=[
                       'wl', 'rel_flux', 'abs_flux', 'rel_flux_conv', 'abs_flux_conv'], skiprows=1)

    wl = df['wl'].values
    flux = df['abs_flux'].values

    if 'marcs_flux_mod' in conf.keys():
        flux *= float(conf['marcs_flux_mod'])
    if 'marcs_wl_mod' in conf.keys():
        wl *= conf['marcs_wl_mod']

    wl = air2vac(wl)

    v = par['radial_velocity']
    c_loc = c * 1e-3
    shift = (1 + v / c_loc) * wl
    flux = np.interp(wl, shift, flux)

    return wl, flux

###
#  LIMB DARKENING
###


def linear(new, old, values):
    # TODO use spline interpolation?
    return np.interp(new, old, values)


def spline(new, old, values):
    return interp1d(old, values, kind='quadratic')(new)


def read(fname, imu, interpolate='linear'):
    """
    reads one limb darkening file and interpolates the values to the given mu values
    fname: filename
    imu: mu values to interpolate to
    interpolate: interpolation function to use
    """

    if interpolate in ['l', 'linear']:
        interpolate = linear
    elif interpolate in ['s', 'spline']:
        interpolate = spline
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

        # TODO am I missing sth here?
        # skip to end of line
        f.read(8)

    result = result[:i]
    return result


def read_all(fname, imu, ld_format, interpolate='linear'):
    """
    read several files in a series
    fname: filename unformated
    imu: range of mu values
    frim: first value to use in fname.format
    to: last value
    interpolate: interpolation method
    """
    result = [read(fname.format(i), imu, interpolate=interpolate)
              for i in ld_format]
    result = np.concatenate(result)
    return result


def load_data(config, par):
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
        data = read_all(flux_file, imu, ld_format, interpolation)
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


def load_intensities(config, par):
    imu = config['star_intensities']
    wl, result = load_data(config, par)

    if 'marcs_flux_mod' in config.keys():
        result *= float(config['marcs_flux_mod'])
    if 'marcs_wl_mod' in config.keys():
        wl *= config['marcs_wl_mod']

    df = pd.DataFrame(result, columns=imu)
    return wl, df

def load_limb_darkening(config, par):
    wl_i, intensities = load_intensities(config, par)
    wl_f, flux = load_flux_directly(config, par)

    flux = np.interp(wl_i, wl_f, flux)
    intensities = intensities.apply(lambda s: s / flux)
    return wl_i, intensities

###
# Solar Model
###

def load_solar(conf, par, calib_dir):
    s_fname = join(calib_dir, 'sun.flx')
    df = pd.read_csv(s_fname, header=None, names=['FLUX'])
    s_flux = df['FLUX'].values

    s_wl_fname = join(calib_dir, 'flx_wavelengths.vac')
    df = pd.read_csv(s_wl_fname, header=None, names=['WAVE'])
    s_wave = df['WAVE'].values

    if 'marcs_flux_mod' in conf.keys():
        s_flux *= float(conf['marcs_flux_mod'])
    if 'marcs_wl_mod' in conf.keys():
        s_wave *= float(conf['marcs_wl_mod'])

    return s_wave, s_flux