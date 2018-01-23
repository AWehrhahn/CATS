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
from awlib.util import normalize
from data_module_interface import data_module
from dataset import dataset as ds
from marcs import marcs
from idl import idl


class harps(data_module):
    """ access HARPS data """
    @classmethod
    def apply_modifiers(cls, conf, par, obs):
        if 'harps_flux_mod' in conf.keys():
            obs.scale *= float(conf['harps_flux_mod'])
        if 'harps_wl_mod' in conf.keys():
            obs.wl *= float(conf['harps_wl_mod'])
        return obs

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
        obs = ds(wave, flux, err)

        obs.wl = air2vac(obs.wl)

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
            obs.doppler_shift(bc_velocity)

        obs = cls.apply_modifiers(conf, par, obs)

        
        obs.phase = phase

        return obs

    @classmethod
    def load_observations(cls, conf, par, *args, **kwargs):
        """ Load all observations from all fits files in the HARPS directory """
        fname = join(conf['input_dir'], conf['harps_dir'],
                     conf['harps_file_obs'])
        wl, obs, err, phase = [], [], [], 
        for g in glob.glob(fname):
            w, f, e, p = cls.load(conf, par, g)

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
        ref = cls.load(conf, par, fname, apply_barycentric=True)
        del ref.phase #We don't need that here
        return ref[1000:]

    @classmethod
    def flux_calibration(cls, conf, par, wl, obs, err, tellurics=True, source='idl', plot=True, plot_title=''):
        calib_dir = join(conf['input_dir'], conf['harps_dir'],
                         conf['harps_calibration_dir'])

        # load harps observation of Vesta (or other object)
        reference = 'Vesta.fits'
        r_wave, r_flux, _ = cls.load_solar(conf, par, reference)
        r_wave = doppler_shift(r_wave, par['radial_velocity'])
        r_flux = interp1d(r_wave, r_flux, kind='quadratic',
                          bounds_error=False, fill_value=0)(wl)
        r_flux = gaussbroad(r_flux, 2)

        if source == 'marcs':
            # load marcs solar spectrum
            s_wave, s_flux = marcs.load_solar(conf, par, calib_dir)
        elif source == 'idl':
            s_wave, s_flux = idl.load_solar(conf, par, calib_dir)

        s_flux = cls.interpolate(wl, s_wave, s_flux)

        # Load telluric spectrum
        t_wave, t_flux = cls.load_tellurics(conf, par)
        t_flux = cls.interpolate(wl, t_wave, t_flux)

        if tellurics:
            s_flux *= t_flux

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

        # TODO get these areas automatically/ from somewhere else
        exclusion = np.array(
            [(5300, 5340), (5850, 6000), (6260, 6340), (6400, 6600), (6860, 7000)])
        tmp = np.zeros((exclusion.shape[0], wl.shape[0]))
        for i, ex in enumerate(exclusion):
            tmp[i] = ~((wl > ex[0]) & (wl < ex[1]))

        tmp = np.all(tmp, axis=0)

        # compare
        profile = np.where(tmp, s_flux / r_flux, 0)
        low, high = min(wl), max(wl)
        for i in range(exclusion.shape[0] + 1):
            if i < exclusion.shape[0]:
                band = (wl >= low) & (wl < exclusion[i, 0])
                low = exclusion[i, 1]
            else:
                band = (wl >= low) & (wl < high)

            profile[band] = gaussbroad(profile[band], 1000, mode='reflect')

        #profile = np.interp(wl, wl[tmp], profile[tmp])
        profile = interp1d(wl[tmp], profile[tmp],
                           kind='quadratic', fill_value='extrapolate')(wl)

        calibrated = obs * profile[None, :]
        calibrated[:, :50] = calibrated[:, 51]

        #Any errors in s_flux and r_flux are broadened away
        c_err = err * profile[None, :]
        c_err[:, :50] = c_err[:, 51]

        if plot:
            import matplotlib.pyplot as plt
            _wl = wl[obs>0.1]
            _s = gaussbroad(s_flux[obs>0.1], 1000)
            _c = gaussbroad(calibrated[0, obs>0.1], 1000)

            #plt.plot(wl, normalize(r_flux), label='reference')
            plt.plot(wl, normalize(obs), label='observation')
            plt.plot(_wl, _s, label='solar')
            plt.plot(_wl, _c, label='calibrated')
            plt.plot(wl, t_flux, label='tellurics')
            plt.plot(wl, profile * 1e4, label='profile')
            plt.xlim([4000, 7000])
            plt.ylim([0, 1.2])
            plt.title(plot_title)
            plt.legend(loc='best')
            plt.show()

        return calibrated, c_err
