"""
Load reduced HARPS observations
"""

from os.path import join
import glob
import numpy as np
import pandas as pd
import astropy.io.fits as fits

import jdcal
from awlib.astro import air2vac, mjd2datetime


def load(conf, par, fname):
    #fname = 'ADP.2015-01-23T22:55:24.657.fits'
    fname = join(conf['input_dir'], conf['harps_dir'], fname)
    hdulist = fits.open(fname)
    data = hdulist[1].data
    header = hdulist[1].header

    wave = data['WAVE'][0, :]
    flux = data['FLUX'][0, :]
    wave = air2vac(wave)
    
    tmid = header['TMID'] #in mjd
    #dtmid = mjd2datetime(tmid) #do I actually need that?

    #phase?
    transit = par['transit'] - jdcal.MJD_0
    period = par['period']
    phase = ((tmid - (transit - period/2)) / period) % 1
    phase = 360 * phase
    
    return wave, flux, phase


def load_observations(conf, par):
    fname = join(conf['input_dir'], conf['harps_dir'], '*.fits')
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

    return wl, obs, phase

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
