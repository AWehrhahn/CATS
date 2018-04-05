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
import joblib

import intermediary as iy
from awlib.astro import air2vac, doppler_shift, planck
from awlib.util import normalize
from awlib.reduce.echelle import rdech
from data_module_interface import data_module
from dataset import dataset
from marcs import marcs
from idl import idl


class harps(data_module):
    """ access HARPS data
    """
    @classmethod
    def apply_modifiers(cls, conf, par, ds):
        """ Apply modifiers as defined in conf

        Parameters:
        ----------
        conf : {dict}
            configuration settings
        par : {dict}
            stellar and planetary parameters
        ds : {dataset}
            dataset to modify
        Returns
        -------
        ds : dataset
            modified dataset
        """

        if 'harps_flux_mod' in conf.keys():
            ds.scale *= float(conf['harps_flux_mod'])
        if 'harps_wl_mod' in conf.keys():
            ds.change_grid(ds.wl * float(conf['harps_wl_mod']))
        return ds

    @classmethod
    def load(cls, conf, par, fname, apply_barycentric=False):
        """ load a single FITS file with HARPS observations

        Assumes that the primary header contains a table with three columns
        WAVE : the wavelength
        FLUX : the spectrum
        ERR : the errors on the spectrum
        as well as TMID in its header, which is the julian date at the middle of the observation

        Parameters:
        ----------
        conf : {dict}
            configuration settings
        par : {dict}
            stellar and planetary parameters
        fname : {str}
            filename, relative to harps directory as defined in conf
        apply_barycentric : {bool}, optional
            apply barycentric correction if True (the default is False)

        Returns
        -------
        obs : dataset
            a single HARPS observation, including the orbital phase
        """
        fname = join(conf['input_dir'], conf['harps_dir'], fname)
        hdulist = fits.open(fname)
        data = hdulist[1].data
        header = hdulist[1].header

        wave = data['WAVE'][0, :]
        flux = data['FLUX'][0, :]
        err = data['ERR'][0, :]

        obs = dataset(wave, flux, err)

        obs.wl = air2vac(obs.wl)

        # calc phases
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
        """ Load all observations from all fits files in the HARPS input directory

        The HARPS input directory is defined in conf

        Parameters:
        ----------
        conf : {dict}
            configuration settings
        par : {dict}
            stellar and planetary parameters
        Returns
        -------
        observation : dataset
            Observations
        """
        cls.log(2, 'HARPS')
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
    def load_stellar_flux(cls, conf, par, *args, **kwargs):
        """ Use out of transit observations to create a stellar spectrum

        Averages multiple seperate observations
        Single observations are not flux calibrated, but rather normalised by their mean value

        Parameters:
        ----------
        conf : {dict}
            configuration settings
        par : {dict}
            stellar and planetary parameters
        Returns
        -------
        flux : dataset
            average stellar flux
        """

        """
        Average observations to get stellar flux
        Requires some observations out of transit
        """
        cls.log(2, 'HARPS')
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
    def load_tellurics(cls, conf, par, *args, **kwargs):
        """ load telluric transmission spectrum

        The spectrum is taken from the ESO SkyCalc online tool
        http://www.eso.org/observing/etc/bin/gen/form?INS.MODE=swspectr+INS.NAME=SKYCALC

        Parameters:
        ----------
        conf : {dict}
            configuration settings
        par : {dict}
            stellar and planetary parameters
        Returns
        -------
        telluric : dataset
            telluric transmission spectrum
        """
        cls.log(2, 'HARPS')
        fname = join(conf['input_dir'], conf['harps_dir'],
                     conf['harps_file_tell'])
        df = pd.read_table(fname, delim_whitespace=True)
        wl = df['wave'].values * 10
        tell = df['tell'].values
        ds = dataset(wl, tell)

        ds = cls.apply_modifiers(conf, par, ds)
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
        """ Calibrate the flux by comparison with a Vesta reference

        The Vesta reference is compared to a normalized solar spectrum from a source defined by source
        The fraction between the two is smoothed and applied as a profile to the observation.
        Some telluric features are removed before this.
        The profile is also adjusted for the temperature of the star, by using Planck's law.

        TODO: telluric features should not be hardcoded into this

        Parameters:
        ----------
        conf : {dict}
            configuration settings
        par : {dict}
            stellar and planetary parameters
        obs : {dataset}
            observation to calibrate
        apply_temp_ratio : {bool}, optional
            apply Planck's law to match temperature difference between the observed star and the sun (the default is True)
        plot : {bool}, optional
            plot the various spectra if True (the default is True)

        Returns
        -------
        calibrated : dataset
            The flux calibrated dataset
        """

        calib_dir = join(conf['input_dir'], conf['harps_dir'],
                         conf['harps_calibration_dir'])

        # load harps observation of Vesta (or other object)
        reference = 'Vesta.fits'
        ref = cls.load_solar(conf, par, reference)
        ref.doppler_shift(par['radial_velocity'])
        ref.wl = obs.wl
        # ref.gaussbroad(2)

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
            return -np.correlate(ref.flux[0], cls.interpolate(shift, obs.wl, solar.flux[0]))[0]

        def func2(x):
            return np.sum(np.abs(gaussbroad(solar.flux[0], x) - ref.flux[0]))

        sol = minimize(func, x0=par['radial_velocity'], method='Nelder-Mead')
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
        sensitivity = np.where(tmp, solar.flux / ref.flux, 0)

        low, high = min(obs.wl), max(obs.wl)
        for i in range(exclusion.shape[0] + 1):
            if i < exclusion.shape[0]:
                band = (obs.wl >= low) & (obs.wl < exclusion[i, 0])
                low = exclusion[i, 1]
            else:
                band = (obs.wl >= low) & (obs.wl < high)
            sensitivity[0, band] = gaussbroad(
                sensitivity[0, band], 1000, mode='reflect')

        sensitivity[0] = cls.interpolate(obs.wl, obs.wl[tmp], sensitivity[0, tmp])

        bbflux = planck(obs.wl, par['t_eff'])  # Teff of the star
        bbflux2 = planck(obs.wl, 5770)  # Teff of the sun
        if apply_temp_ratio:
            # Fix Difference between solar and star temperatures
            ratio = bbflux2 / bbflux
        else:
            ratio = 1

        # Apply changes
        calibrated = obs.flux * sensitivity * ratio
        calibrated[:, :50] = calibrated[:, 51, None]

        # Any errors in s_flux and r_flux are broadened away ?
        c_err = obs.err * sensitivity * ratio
        c_err[:, :50] = c_err[:, 51, None]

        distance = 1 / (1e-3 * par['parallax'])
        distance_modulus = 800/6

        calibrated *= distance_modulus
        calibrated = dataset(obs.wl, calibrated, c_err)

        name = 'correction.pkl'
        func = sensitivity * ratio * distance_modulus
        joblib.dump([obs.wl, func], name)

        if plot:

            calib_dir = join(conf['input_dir'], conf['marcs_dir'])
            #comparison = marcs.load_simple_data(conf, par, fname='comparison.flx')
            

            import matplotlib.pyplot as plt
            import matplotlib.transforms as mtransforms
            #import plotly.offline as py
            #from awlib.pltly import pltly
            #plt = pltly()

            # calibrated.gaussbroad(50)
            # solar.gaussbroad(50)

            #plt.plot(wl, normalize(r_flux), label='reference')
            wl = obs.wl
            if obs.flux.ndim == 1:
                _flux = obs.flux
            else:
                _flux = obs.flux[0]

            #comparison.wl = wl

            fig, ax = plt.subplots()
            trans = mtransforms.blended_transform_factory(
                ax.transData, ax.transAxes)
            ax.fill_between(wl, 0, 1.2, where=~tmp,
                            facecolor='green', alpha=0.5, transform=trans)

            #ax.plot(wl, solar.flux, label='solar', color='tab:blue')
            #plt.plot(wl, comparison.flux, label='comparison', color='tab:blue')
            plt.plot(wl, _flux,
                     label='observation', color='tab:green')
            ax.plot(wl, ref.flux[0],
                    label='reference', color='tab:pink')

            plt.plot(wl,
                     calibrated.flux[0], label='calibrated', color='tab:orange')
            #plt.plot(wl, tell.flux, label='tellurics', color='tab:green')
            #ax.plot(wl, bbflux / max(bbflux), label='4000 K', color='tab:pink')
            #ax.plot(wl, bbflux2 / max(bbflux2), label='5770 K', color='tab:purple')
            ax.plot(wl, sensitivity[0] * 1e4,
                    label='sensitivity', color='tab:red')
            #plt.plot(wl, ratio / ratio.max(), label='ratio')
            plt.xlim([4700, 5000])
            plt.ylim([0, 1.2])
            plt.title(plot_title)
            plt.legend(loc='best')
            # py.plot_mpl(plt.gcf())
            plt.show()

        return calibrated

    @classmethod
    def load_reduced(cls, conf, par, no_cont=False):
        """Load a reduced echelle file

        It is assumed that the reduced .ech file has also been wavelength calibrated

        Parameters:
        ----------
        conf : {dict}
            configuration settings
        par : {dict}
            stellar parameters
        no_cont : {bool}, optional
            wether to apply the continuum (the default is False, which [default_description])

        Returns
        -------
        dataset
            the observation
        """

        fname = 'HARPS.2016-04-09T01:55:25.400c.ech'
        fname = join(conf['input_dir'], conf['harps_dir'], fname)

        ech = rdech(fname)
        
        wave = ech.wave.reshape(-1)
        sort = np.argsort(wave)

        wave = wave[sort]
        spec = ech.spec.reshape(-1)[sort]
        sig = ech.sig.reshape(-1)[sort]

        ds = dataset(wave, spec, sig)
        return ds
