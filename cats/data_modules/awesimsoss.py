import logging
import os.path
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as q
from scipy.interpolate import interp2d
from scipy.constants import h, c, k

from .data_interface import data_raw

try:
    from awesimsoss import TSO, STAR_DATA, PLANET_DATA
    from awesimsoss.utils import wave_solutions
    from awesimsoss.make_trace import trace_polynomials
    import batman

    hasAwesimsoss = True
except ImportError:
    hasAwesimsoss = False
finally:
    if not hasAwesimsoss:
        logging.error("Could not import awesimsoss module")


class awesimsoss(data_raw):

    _requires = ["parameters", "planet", "stellar_flux"]

    def bbr(self, wave, teff):
        wave = wave.to_value(q.meter)
        tmp = np.exp((h * c) / (wave * k * teff)) - 1
        flux = 2 * h * c ** 2 * wave ** (-5) * tmp ** (-1)
        flux = flux << (q.Joule / (q.m ** 3 * q.s))
        flux = flux.to(q.erg / (q.AA * q.cm ** 2 * q.s))

        flux *= STAR_DATA[1].max().value / flux.max().value

        return flux

    def create_observation(self, star, planet):

        parameters = self.parameters
        # Set up simulation object
        name = self.configuration["_star"] + self.configuration["_planet"]
        ngrps = self.configuration["ngroups"]
        nints = self.configuration["nintegrations"]
        tso_transit = TSO(ngrps=ngrps, nints=nints, star=star, target=name)

        # Set up transit parameters
        params = batman.TransitParams()
        # parameters["periastron"] # time of inferior conjunction
        params.t0 = 0
        # orbital period (days)
        params.per = parameters["period"].to_value("day")
        # semi-major axis (in units of stellar radii)
        params.a = parameters["sma"].to_value(q.R_sun)
        # radius ratio for Jupiter orbiting the Sun
        params.rp = (parameters["r_planet"] / parameters["r_star"]).to_value("")
        # orbital inclination (in degrees)
        params.inc = parameters["inc"].to_value("deg")
        # eccentricity
        params.ecc = parameters["ecc"].to_value("")
        # longitude of periastron (in degrees) p
        params.w = parameters["w"].to_value("deg")
        # limb darkening profile to use
        params.limb_dark = "quadratic"
        # limb darkening coefficients
        params.u = [0.1, 0.1]

        tmodel = batman.TransitModel(params, tso_transit.time)
        # effective temperature of the host star
        tmodel.teff = parameters["teff"].to_value("K")
        # log surface gravity of the host star
        tmodel.logg = parameters["logg"].to_value("")
        # metallicity of the host star
        tmodel.feh = parameters["monh"].to_value("")

        tso_transit.simulate(planet=planet, tmodel=tmodel)

        folder = self.configuration["dir"]
        fname = self.configuration["fname"]
        fname = os.path.join(folder, fname)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        tso_transit.to_fits(fname)
        return fname

    def create_bias(self):
        star = [STAR_DATA[0], np.zeros_like(STAR_DATA[1])]
        tso_transit = TSO(ngrps=1, nints=1, star=star)
        tso_transit.simulate()

        folder = self.configuration["dir"]
        fname = "bias.fits"
        fname = os.path.join(folder, fname)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        tso_transit.to_fits(fname)
        return fname

    def create_flat(self):
        teff = self.parameters["teff"].to_value("K")
        wave = STAR_DATA[0]
        flux = self.bbr(wave, teff)
        star = [STAR_DATA[0], flux]

        tso_transit = TSO(ngrps=1, nints=1, star=star)
        tso_transit.simulate()

        folder = self.configuration["dir"]
        fname = "flat.fits"
        fname = os.path.join(folder, fname)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        tso_transit.to_fits(fname)
        return fname

    def create_wavelength(self):
        subarray = "SUBSTRIP256"
        x = np.arange(4, 2045)
        y = np.arange(256)
        out = np.zeros((2, 2041))

        for order in [1, 2]:
            wave = wave_solutions(subarray, order)
            wave = wave[:, 4:2045]
            coeffs = trace_polynomials(subarray, order)
            trace = np.polyval(coeffs, x)
            wave = interp2d(x, y, wave)

            for i, j in zip(x, trace):
                out[order - 1, i - x[0]] = wave(i, j)[0]

        out *= q.micrometer.to(q.AA)

        folder = self.configuration["dir"]
        fname = "jwst_niriss_gr700xd.thar.npz"
        fname = os.path.join(folder, "../reduced", fname)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        np.savez(fname, wave=out, thar=0, linelist=0, coef=0)

    def get_raw(self, **data):
        self.parameters = data["parameters"]

        planet = data["planet"]
        star = data["stellar_flux"]
        flux_unit = q.erg / (q.Angstrom * q.cm**2 * q.s)
        planet = [planet.wave, planet.data]
        star = [star.wave << q.Angstrom, star.data << flux_unit]
        star[0] = star[0].to(q.micrometer)
        star[1] *= STAR_DATA[1].max().value / star[1].max().value
        # TODO find a better way to create a planet spectrum from the transition spectrum
        planet[0] *= q.AA.to(q.micrometer)
        planet[1] = 1 - planet[1]
        planet[1] *= (PLANET_DATA[1].mean() - PLANET_DATA[1].min()) / planet[1].max()
        planet[1] += PLANET_DATA[1].min()

        # self.create_wavelength()
        # # Create empty detector image
        # self.create_bias()
        # # Create image with black body "star"
        # self.create_flat()
        # # Create the actual observation
        # self.create_observation(star, planet)
        
        folder = self.configuration["dir"]
        target = self.configuration["_star"] + self.configuration["_planet"]
        
        return "JWST_NIRISS", "GR700XD", target, folder
