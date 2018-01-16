"""
Load reduced HARPS observations
"""

from os.path import join
import glob
import numpy as np
import pandas as pd
import astropy.io.fits as fits
from scipy.signal import correlate
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d as gaussbroad

import jdcal
from awlib.astro import air2vac, doppler_shift

import marcs


def load(conf, par, fname, apply_barycentric=False):
    """ load a single fits file in the HARPS directory """
    fname = join(conf['input_dir'], conf['harps_dir'], fname)
    hdulist = fits.open(fname)
    data = hdulist[1].data
    header = hdulist[1].header

    wave = data['WAVE'][0, :]
    flux = data['FLUX'][0, :]
    wave = air2vac(wave)

    tmid = header['TMID']  # in mjd
    # dtmid = mjd2datetime(tmid) #do I actually need that?

    # phase?
    transit = par['transit'] - jdcal.MJD_0
    period = par['period']
    phase = ((tmid - (transit - period / 2)) / period) % 1
    phase = 360 * phase

    # barycentric velocity
    if apply_barycentric:
        bc_velocity = -hdulist[0].header['ESO DRS BERV']
        flux = doppler_shift(wave, flux, bc_velocity)

    return wave, flux, phase


def load_observations(conf, par):
    """ Load all observations from all fits files in the HARPS directory """
    fname = join(conf['input_dir'], conf['harps_dir'], conf['harps_file_obs'])
    wl, obs, phase = [], [], []
    for g in glob.glob(fname):
        w, f, p = load(conf, par, g)

        wl.append(w)
        f = np.interp(wl[0], w, f)
        wl[-1] = wl[0]
        obs.append(f)
        phase.append(p)

    wl = np.array(wl)
    obs = np.array(obs)
    phase = np.array(phase)
    phase = np.deg2rad(phase)

    return wl[0], obs, phase


def load_flux(conf, par):
    """
    Average observations to get stellar flux
    Requires some observations out of transit
    """
    wl, flux, phase = load_observations(conf, par)
    # Don't use observations during transit
    flux = flux[(phase > 181) | (phase < 179)]
    total = np.mean(flux)
    avg = np.mean(flux, 1)
    flux = flux * total / avg[:, None]
    wl = wl[0]
    flux = np.mean(flux, 0)
    return wl, flux


def load_tellurics(conf, par):
    """
    load telluric data from skycalc
    http://www.eso.org/observing/etc/bin/gen/form?INS.MODE=swspectr+INS.NAME=SKYCALC
    """
    fname = join(conf['input_dir'], conf['harps_dir'], conf['harps_file_tell'])
    df = pd.read_table(fname, delim_whitespace=True)
    wl = df['wave']
    tell = df['tell']

    if 'harps_flux_mod' in conf.keys():
        tell *= float(conf['harps_flux_mod'])
    if 'harps_wl_mod' in conf.keys():
        wl *= float(conf['harps_wl_mod'])

    return wl, tell


def load_solar(conf, par, reference='Vesta.fits'):
    """ load the HARPS reflected solar spectrum """
    fname = join(conf['input_dir'], conf['harps_dir'],
                 conf['harps_calibration_dir'], reference)
    r_wave, r_flux, _ = load(conf, par, fname, apply_barycentric=True)
    return r_wave, r_flux


def flux_calibration(conf, par, wl, obs):
    calib_dir = join(conf['input_dir'], conf['harps_dir'],
                     conf['harps_calibration_dir'])

    # TODO automatically find best offset and broadening
    # Use Cross Correlation?
    # load harps observation of Vesta (or other object)
    reference = 'Vesta.fits'
    r_wave, r_flux = load_solar(conf, par, reference)
    #r_flux = doppler_shift(r_wave, r_flux, par['radial_velocity'])
    r_flux = interp1d(r_wave, r_flux, kind='quadratic',
                      bounds_error=False, fill_value=0)(wl)
    #r_flux = gaussbroad(r_flux, sigma = 2)

    # load marcs solar spectrum
    s_wave, s_flux = marcs.load_solar(conf, par, calib_dir)
    s_flux = interp1d(s_wave, s_flux, kind='quadratic')(wl)
    
    t_wave, t_flux = load_tellurics(conf, par)
    t_flux = interp1d(t_wave, t_flux, kind='quadratic', bounds_error=False)(wl)
    #t_flux = gaussbroad(t_flux, 2)

    s_flux *= t_flux
    s_flux = gaussbroad(s_flux, 4)


    def func(x):
        #I would split this for readability if that wouldn't take too much time
        return -np.correlate(s_flux, gaussbroad(doppler_shift(wl, r_flux, x[0], 'linear'), abs(x[1])))[0]

    sol = minimize(func, x0=[31.45, 2])
    v, b = sol.x
    r_flux = doppler_shift(wl, r_flux, v)
    r_flux = gaussbroad(r_flux, b)
    #s_flux = gaussbroad(s_flux, b2)

    # compare
    profile = np.where(s_flux != 0, r_flux / s_flux, 0)
    profile = gaussbroad(profile, 1000)
    calibrated = np.where(profile[None, :] > 0.2, obs / profile[None, :], 0)

    """
    # TODO
    import matplotlib.pyplot as plt
    plt.plot(wl, s_flux, label='solar')
    plt.plot(wl, r_flux, label='reference')
    plt.plot(wl, obs * 100, label='observation')
    plt.plot(wl, profile * 100000, label='profile')
    #plt.plot(wl, calibrated[0] * 100, label='calibrated')
    #plt.xlim([5168, 5168.4])
    plt.ylim([0, 2e5])
    plt.legend(loc='best')
    plt.show()
    """
    return calibrated
