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
from awlib.astro import air2vac, doppler_shift, planck
from awlib.util import normalize
from data_module_interface import data_module
from dataset import dataset as ds
from marcs import marcs
from idl import idl


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
        err = data['ERR'][0, :]
        #TODO
        obs = ds(wave, flux, err)

        wave = air2vac(wave)

        tmid = header['TMID']  # in mjd
        transit = par['transit'] - jdcal.MJD_0
        period = par['period']
        phase = ((tmid - (transit - period / 2)) / period) % 1
        phase = 360 * phase

        # barycentric velocity
        if apply_barycentric:
            bc_velocity = -hdulist[0].header['ESO DRS BERV']
            shift = doppler_shift(wave, bc_velocity)
            flux = cls.interpolate(wave, shift, flux)

        wave, flux = cls.apply_modifiers(conf, par, wave, flux)

        obs.phase = phase

        return wave, flux, err, phase

    @classmethod
    def load_observations(cls, conf, par, *args, **kwargs):
        """ Load all observations from all fits files in the HARPS directory """
        fname = join(conf['input_dir'], conf['harps_dir'],
                     conf['harps_file_obs'])
        wl, obs, err, phase = [], [], [], []
        for g in glob.glob(fname):
            w, f, e, p = cls.load(conf, par, g)

            wl.append(w)
            f = np.interp(wl[0], w, f)
            wl[-1] = wl[0]
            obs.append(f)
            phase.append(p)
            err.append(e)

        wl = np.array(wl)
        obs = np.array(obs)
        err = np.array(err)
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
        r_wave, r_flux, r_err, _ = cls.load(conf, par, fname, apply_barycentric=True)
        #del ref.phase #We don't need that here
        return r_wave[1000:], r_flux[1000:], r_err[1000:]

    @classmethod
    def flux_calibration(cls, conf, par, wl, obs, err, source='idl', plot=True, plot_title=''):
        calib_dir = join(conf['input_dir'], conf['harps_dir'],
                         conf['harps_calibration_dir'])

        # load harps observation of Vesta (or other object)
        reference = 'Vesta.fits'
        r_wave, r_flux, _ = cls.load_solar(conf, par, reference)
        r_wave = doppler_shift(r_wave, par['radial_velocity'])
        r_flux = cls.interpolate(wl, r_wave, r_flux)
        r_flux = gaussbroad(r_flux, 2)

        if source == 'marcs':
            # load marcs solar spectrum
            tellurics = True
            s_wave, s_flux = marcs.load_solar(conf, par, calib_dir)
        elif source == 'idl':
            tellurics = False
            s_wave, s_flux = idl.load_solar(conf, par, calib_dir)

        s_flux = cls.interpolate(wl, s_wave, s_flux)

        # Load telluric spectrum
        t_wave, t_flux = cls.load_tellurics(conf, par)
        t_flux = cls.interpolate(wl, t_wave, t_flux)

        if tellurics:
            s_flux *= t_flux

        ###
        # Fix radial velocity and broadening of the solar spectrum
        ###

        def func(x):
            # also fitting for best broadening at the same time doesn't work
            shift = doppler_shift(wl, x)
            return -np.correlate(r_flux, cls.interpolate(wl, shift, s_flux))[0]

        def func2(x):
            return np.sum(np.abs(gaussbroad(s_flux, x) - r_flux))

        sol = minimize(func, x0=par['radial_velocity'])
        v = sol.x[0]
        print('shift: ', v)
        shift = doppler_shift(wl, v)
        s_flux = cls.interpolate(wl, shift, s_flux)

        sol = minimize(func2, x0=1)
        # the fit wants to make the solar spectrum broader than it needs to be
        b = np.abs(sol.x[0] - 1)
        print('broadening: ', b)
        if b != 0:
            s_flux = gaussbroad(s_flux, b)

        ###
        # Create broadened profile
        ###

        # Define Exclusion areas manually, usually telluric line
        # TODO get these areas automatically/from somewhere else
        exclusion = np.array(
            [(5300, 5340), (5850, 6000), (6260, 6340), (6400, 6600), (6860, 7000)])
        tmp = np.zeros((exclusion.shape[0], wl.shape[0]))
        for i, ex in enumerate(exclusion):
            tmp[i] = ~((wl > ex[0]) & (wl < ex[1]))
        tmp = np.all(tmp, axis=0)

        # be careful to only broaden within individual sections
        profile = np.where(tmp, s_flux / r_flux, 0)
        low, high = min(wl), max(wl)
        for i in range(exclusion.shape[0] + 1):
            if i < exclusion.shape[0]:
                band = (wl >= low) & (wl < exclusion[i, 0])
                low = exclusion[i, 1]
            else:
                band = (wl >= low) & (wl < high)
            profile[band] = gaussbroad(profile[band], 1000, mode='reflect')

        profile = cls.interpolate(wl, wl[tmp], profile[tmp])


        #Fix Difference between solar and star temperatures
        bbflux = planck(wl, 4000) # Teff of the star
        bbflux2 = planck(wl, 6770) # Teff of the sun
        ratio = bbflux2 / bbflux

        #Apply changes
        calibrated = obs * profile[None, :] * ratio
        calibrated[:, :50] = calibrated[:, 51]

        #Any errors in s_flux and r_flux are broadened away
        c_err = err * profile[None, :]
        c_err[:, :50] = c_err[:, 51]

        calibrated[0] *= 10

        if plot:
            import matplotlib.pyplot as plt
            calibrated[0] = gaussbroad(calibrated[0], 50)
            s_flux = gaussbroad(s_flux, 50)

            #plt.plot(wl, normalize(r_flux), label='reference')
            plt.plot(wl, normalize(obs), label='observation')
            plt.plot(wl, s_flux, label='solar')
            plt.plot(wl, calibrated[0], label='calibrated')
            plt.plot(wl, t_flux, label='tellurics')
            plt.plot(wl, profile * 1e4, label='profile')
            plt.plot(wl, ratio, label='ratio')
            plt.xlim([4000, 7000])
            plt.ylim([0, 1.2])
            plt.title(plot_title)
            plt.legend(loc='best')
            plt.show()

        return calibrated, c_err
