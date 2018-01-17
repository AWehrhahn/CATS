"""
Load reduced HARPS observations
"""

import glob
from os.path import join

import astropy.io.fits as fits
import jdcal
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d as gaussbroad
from scipy.optimize import minimize

import intermediary as iy
from awlib.astro import air2vac, doppler_shift
from data_module_interface import data_module
from marcs import marcs


class harps(data_module):
    """ access HARPS data """
    @classmethod
    def apply_modifiers(cls, conf, par, wl, flux):
        if 'harps_flux_mod' in conf.keys():
            flux *= float(conf['harps_flux_mod'])
        if 'harps_wl_mod' in conf.keys():
            wl *= float(conf['harps_wl_mod'])
        return wl, flux

    @classmethod
    def load(cls, conf, par, fname, apply_barycentric=False):
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

        wave, flux = cls.apply_modifiers(conf, par, wave, flux)

        return wave, flux, phase

    @classmethod
    def load_observations(cls, conf, par, *args, **kwargs):
        """ Load all observations from all fits files in the HARPS directory """
        fname = join(conf['input_dir'], conf['harps_dir'],
                     conf['harps_file_obs'])
        wl, obs, phase = [], [], []
        for g in glob.glob(fname):
            w, f, p = cls.load(conf, par, g)

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

    @classmethod
    def load_stellar_flux(cls, conf, par):
        """
        Average observations to get stellar flux
        Requires some observations out of transit
        """
        wl, flux, phase = cls.load_observations(conf, par)
        # Don't use observations during transit
        flux = flux[(phase > np.pi + iy.maximum_phase(par)) |
                    (phase < np.pi - iy.maximum_phase(par))]
        total = np.mean(flux)
        avg = np.mean(flux, 1)
        flux = flux * total / avg[:, None]
        flux = np.mean(flux, 0)
        return wl, flux

    @classmethod
    def load_tellurics(cls, conf, par):
        """
        load telluric data from skycalc
        http://www.eso.org/observing/etc/bin/gen/form?INS.MODE=swspectr+INS.NAME=SKYCALC
        """
        fname = join(conf['input_dir'], conf['harps_dir'],
                     conf['harps_file_tell'])
        df = pd.read_table(fname, delim_whitespace=True)
        wl = df['wave']
        tell = df['tell']

        wl, tell = cls.apply_modifiers(conf, par, wl, tell)
        wl *= 10  # TODO only tellurics has this shify
        return wl, tell

    @classmethod
    def load_solar(cls, conf, par, reference='Vesta.fits'):
        """ load the HARPS reflected solar spectrum """
        fname = join(conf['input_dir'], conf['harps_dir'],
                     conf['harps_calibration_dir'], reference)
        r_wave, r_flux, _ = cls.load(conf, par, fname, apply_barycentric=True)
        return r_wave, r_flux

    @classmethod
    def flux_calibration(cls, conf, par, wl, obs):
        calib_dir = join(conf['input_dir'], conf['harps_dir'],
                         conf['harps_calibration_dir'])

        # load harps observation of Vesta (or other object)
        reference = 'Vesta.fits'
        r_wave, r_flux = cls.load_solar(conf, par, reference)
        r_flux = doppler_shift(r_wave, r_flux, par['radial_velocity'])
        r_flux = interp1d(r_wave, r_flux, kind='quadratic',
                          bounds_error=False, fill_value=0)(wl)
        #r_flux = gaussbroad(r_flux, sigma = 4)

        # load marcs solar spectrum
        s_wave, s_flux = marcs.load_solar(conf, par, calib_dir)
        s_flux = interp1d(s_wave, s_flux, kind='quadratic',
                          fill_value=np.nan, bounds_error=False)(wl)

        t_wave, t_flux = cls.load_tellurics(conf, par)
        t_flux = interp1d(t_wave, t_flux, kind='quadratic',
                          bounds_error=False)(wl)

        s_flux *= t_flux
        #s_flux = gaussbroad(s_flux, 4)

        def func(x):
            # also fitting for best broadening doesn't work
            return -np.correlate(s_flux, doppler_shift(wl, r_flux, x, 'linear'))[0]

        def func2(x):
            return np.sum(np.abs(gaussbroad(s_flux, x) - r_flux))

        sol = minimize(func, x0=par['radial_velocity'])
        v = sol.x[0]
        print('shift: ', v)
        #r_flux = doppler_shift(wl, r_flux, v)
        s_flux = doppler_shift(wl, s_flux, -v)

        sol = minimize(func2, x0=1)
        b = sol.x[0]
        print('broadening: ', b)
        s_flux = gaussbroad(s_flux, b)

        # compare
        profile = np.where(s_flux != 0, r_flux / s_flux, 0)
        profile = gaussbroad(profile, 1000)
        calibrated = np.where(
            profile[None, :] > 0.2, obs / profile[None, :], 0)

        # TODO
        import matplotlib.pyplot as plt
        plt.plot(wl, s_flux, label='solar')
        plt.plot(wl, r_flux, label='reference')
        plt.plot(wl, obs, label='observation')
        plt.plot(wl, profile * 100000, label='profile')
        plt.plot(wl, calibrated[0], label='calibrated')
        plt.xlim([4309, 4314])
        plt.ylim([0, 1.25e5])
        plt.legend(loc='best')
        plt.show()

        return calibrated
