"""
Load MARCS model stellar spectra
marcs.astro.uu.se
"""
import os
from os.path import join

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.constants import c, pi

from DataSources import Cache

###
#  STELLAR FLUX
###


def scaling_factor(par):
    dist = 1 / (par['parallax'] * 1e-3) * 3.086e16  # in m
    h = 10**2  # m**2 #Telescope size
    return h / dist**2


def air2vac(wl_air):
    ii = np.where(wl_air > 1999.352)
    #ii = where(wl_air gt 1999.352, nii)
    wl_vac = wl_air

    sigma2 = (1e4 / wl_air[ii])**2  # Compute wavenumbers squared
    fact = 1e0 + 8.336624212083e-5                            \
            + 2.408926869968e-2 / (1.301065924522e2 - sigma2) \
            + 1.599740894897e-4 / (3.892568793293e1 - sigma2)
    wl_vac[ii] = wl_air[ii] * fact  # Convert to vacuum wavelength

    return wl_vac



def load_flux(config, par):
    """ load MARCS flux files """
    wl, data = load_data(config, par)
    
    imu = config['star_intensities']
    intensities = data

    r = np.sqrt(1-imu**2)
    i_tmp = [intensities[:, 0] * np.pi * r[0]**2]
    for j in range(1, len(imu)):
        i_tmp.append(intensities[:, j] * np.pi * (r[j]**2 - r[j-1]**2))
    
    flux = np.sum(i_tmp, 0)/np.pi

    if 'marcs_flux_mod' in config.keys():
        flux *= float(config['marcs_flux_mod'])
    if 'marcs_wl_mod' in config.keys():
        wl *= config['marcs_wl_mod']

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
    result = [read(fname.format(i), imu, interpolate=interpolate) for i in ld_format]
    result = np.concatenate(result)
    return result

def load_data(config, par):
    # filename
    flux_file = join(config['input_dir'], config['marcs_dir'], config['marcs_file_ld'])
    imu = config['star_intensities']
    ld_format = config['marcs_ld_format']
    interpolation = config['marcs_ld_interpolation_method']
    cache = Cache.Cache(config['path_cache'], par['name_star'], par['name_planet'], flux_file, ld_format, imu, interpolation)

    try:
        data = cache.load()
    except OSError:
        data = None

    if data is None:
        data = read_all(flux_file, imu, ld_format, interpolation)
        cache.save(data)

    # plt.plot(data[:, 0], data[:, 1])

    # Doppler shift
    wl = data[:, 0]

    wl = air2vac(wl)

    v = par['radial_velocity']
    c_loc = c * 1e-3
    shift = (1 + v/c_loc) * wl
    for i in range(1, data.shape[1]):
        data[:, i] = np.interp(wl, shift, data[:, i])

    # plt.plot(data[:, 0], data[:, 1])
    # plt.show()

    return data[:, 0], data[:, 1:]

def load_limb_darkening(config, par):
    imu = config['star_intensities']
    wl, result = load_data(config, par)

    if 'marcs_flux_mod' in config.keys():
        result *= float(config['marcs_flux_mod'])
    if 'marcs_wl_mod' in config.keys():
        wl *= config['marcs_wl_mod']

    df = pd.DataFrame(result, columns=imu)
    return wl, df
