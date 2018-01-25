"""
Load reduced HARPS observations
"""

import glob
from os.path import join

import astropy.io.fits as fits
import jdcal
import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d as gaussbroad
from scipy.optimize import minimize

import intermediary as iy
from awlib.astro import air2vac, doppler_shift, planck
from awlib.util import normalize
from data_module_interface import data_module
from dataset import dataset
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

        obs = dataset(wave, flux, err)

        obs.wl = air2vac(obs.wl)

        #calc phases
        tmid = header['TMID']  # in mjd
        transit = par['transit'] - jdcal.MJD_0
        period = par['period']
        phase = ((tmid - (transit - period / 2)) / period) % 1
        phase = 360 * phase

        # barycentric velocity
        if apply_barycentric:
            bc_velocity = -hdulist[0].header['ESO DRS BERV']
            obs.doppler_shift(bc_velocity)

        obs = cls.apply_modifiers(conf, par, obs)
        obs.phase = np.deg2rad(phase)

        return obs

    @classmethod
    def load_observations(cls, conf, par, *args, **kwargs):
        """ Load all observations from all fits files in the HARPS directory """
        fname = join(conf['input_dir'], conf['harps_dir'],
                     conf['harps_file_obs'])

        # Load data
        obs = [cls.load(conf, par, g) for g in glob.glob(fname)]

        # Fix wl grid
        for i in range(1, len(obs)):
            obs[i].wl = obs[0].wl

        # Organize everything into a single dataset
        flux = np.array([ob.flux for ob in obs])
        err = np.array([ob.err for ob in obs])
        phase = np.array([ob.phase for ob in obs])

        obs = dataset(obs[0].wl, flux, err)
        obs.phase = phase

        return obs

    @classmethod
    def load_stellar_flux(cls, conf, par):
        """
        Average observations to get stellar flux
        Requires some observations out of transit
        """
        obs = cls.load_observations(conf, par)
        # Don't use observations during transit
        obs.flux = obs.flux[(obs.phase > np.pi + iy.maximum_phase(par)) |
                    (obs.phase < np.pi - iy.maximum_phase(par))]
        total = np.mean(obs.flux)
        avg = np.mean(obs.flux, 1)
        obs.flux = obs.flux * total / avg[:, None]
        obs.flux = np.mean(obs.flux, 0)
        return obs

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
        ds = dataset(wl, tell)

        ds = cls.apply_modifiers(conf, par, ds)
        ds.wl *= 10  # TODO only tellurics has this shify
        return ds

    @classmethod
    def load_solar(cls, conf, par, reference='Vesta.fits'):
        """ load the HARPS reflected solar spectrum """
        fname = join(conf['input_dir'], conf['harps_dir'],
                     conf['harps_calibration_dir'], reference)
        ref = cls.load(conf, par, fname, apply_barycentric=True)
        # del ref.phase #We don't need that here
        return ref[1000:]

    @classmethod
    def flux_calibration(cls, conf, par, obs, apply_temp_ratio=True, source='idl', plot=True, plot_title=''):
        calib_dir = join(conf['input_dir'], conf['harps_dir'],
                         conf['harps_calibration_dir'])

        # load harps observation of Vesta (or other object)
        reference = 'Vesta.fits'
        ref = cls.load_solar(conf, par, reference)
        ref.doppler_shift(par['radial_velocity'])
        ref.wl = obs.wl
        ref.flux = gaussbroad(ref.flux, 2)

        if source == 'marcs':
            # load marcs solar spectrum
            tellurics = True
            solar = marcs.load_solar(conf, par, calib_dir)
        elif source == 'idl':
            tellurics = False
            solar = idl.load_solar(conf, par, calib_dir)

        solar.wl = obs.wl

        # Load telluric spectrum
        tell = cls.load_tellurics(conf, par)
        tell.wl = obs.wl

        if tellurics:
            solar.flux *= tell.flux

        ###
        # Fix radial velocity and broadening of the solar spectrum
        ###

        def func(x):
            # also fitting for best broadening at the same time doesn't work
            shift = doppler_shift(obs.wl, x)
            return -np.correlate(ref.flux, cls.interpolate(shift, obs.wl, solar.flux))[0]

        def func2(x):
            return np.sum(np.abs(gaussbroad(solar.flux, x) - ref.flux))

        sol = minimize(func, x0=par['radial_velocity'])
        v = sol.x[0]
        print('shift: ', v)
        solar.doppler_shift(v)

        sol = minimize(func2, x0=1)
        # the fit wants to make the solar spectrum broader than it needs to be
        b = np.abs(sol.x[0] - 1)
        print('broadening: ', b)
        if b != 0:
            solar.flux = gaussbroad(solar.flux, b)

        ###
        # Create broadened profile
        ###

        # Define Exclusion areas manually, usually telluric line
        # TODO get these areas automatically/from somewhere else
        exclusion = np.array(
            [(5300, 5340), (5850, 6000), (6260, 6340), (6400, 6600), (6860, 7000)])
        tmp = np.zeros((exclusion.shape[0], obs.wl.shape[0]))
        for i, ex in enumerate(exclusion):
            tmp[i] = ~((obs.wl > ex[0]) & (obs.wl < ex[1]))
        tmp = np.all(tmp, axis=0)

        # be careful to only broaden within individual sections
        profile = np.where(tmp, solar.flux / ref.flux, 0)
        low, high = min(obs.wl), max(obs.wl)
        for i in range(exclusion.shape[0] + 1):
            if i < exclusion.shape[0]:
                band = (obs.wl >= low) & (obs.wl < exclusion[i, 0])
                low = exclusion[i, 1]
            else:
                band = (obs.wl >= low) & (obs.wl < high)
            profile[band] = gaussbroad(profile[band], 1000, mode='reflect')

        profile = cls.interpolate(obs.wl, obs.wl[tmp], profile[tmp])

        if apply_temp_ratio:
            # Fix Difference between solar and star temperatures
            bbflux = planck(obs.wl, 4000)  # Teff of the star
            bbflux2 = planck(obs.wl, 6770)  # Teff of the sun
            ratio = bbflux2 / bbflux * 10 #TODO the factor 10 doesn't belong here anyway
        else:
            ratio = 1

        # Apply changes
        calibrated = obs.flux * profile[None, :] * ratio
        calibrated[:, :50] = calibrated[:, 51, None]

        # Any errors in s_flux and r_flux are broadened away
        c_err = obs.err * profile[None, :]
        c_err[:, :50] = c_err[:, 51, None]

        calibrated = dataset(obs.wl, calibrated, c_err)

        if plot:
            import matplotlib.pyplot as plt
            calibrated.flux = gaussbroad(calibrated.flux, 50)
            solar.flux = gaussbroad(solar.flux, 50)

            #plt.plot(wl, normalize(r_flux), label='reference')
            wl = obs.wl
            if obs.flux.ndim == 1:
                _flux = obs.flux
            else:
                _flux = obs.flux[0]
            plt.plot(wl, normalize(_flux), label='observation')
            plt.plot(wl, solar.flux, label='solar')
            plt.plot(wl, calibrated.flux[0], label='calibrated')
            plt.plot(wl, tell.flux, label='tellurics')
            plt.plot(wl, profile * 1e4, label='profile')
            #plt.plot(wl, ratio, label='ratio')
            plt.xlim([4000, 7000])
            plt.ylim([0, 1.2])
            plt.title(plot_title)
            plt.legend(loc='best')
            plt.show()

        return calibrated