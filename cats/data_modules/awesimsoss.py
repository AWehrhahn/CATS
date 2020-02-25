import logging
import os.path
from itertools import product

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as q
from scipy.interpolate import interp2d
from scipy.constants import h, c, k

from .data_interface import data_raw, data_planet, data_stellarflux
from .dataset import dataset
from ..orbit import Orbit
from ..utils import std_out_err_redirect_tqdm

try:
    from awesimsoss import TSO, STAR_DATA, PLANET_DATA
    from awesimsoss.utils import wave_solutions, subarray
    from awesimsoss.make_trace import trace_polynomials
    import batman

    hasAwesimsoss = True
except ImportError:
    hasAwesimsoss = False
finally:
    if not hasAwesimsoss:
        logging.error("Could not import awesimsoss module")





class awesimsoss(data_raw, data_planet, data_stellarflux):

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

        name = self.configuration["_star"] + self.configuration["_planet"]
        ngrps = self.configuration["ngroups"]
        nints = self.configuration["nintegrations"]
        nexposures = self.configuration["nexposures"]
        tframe = subarray("SUBSTRIP256")["tfrm"]

        folder = self.configuration["dir"]
        fname = self.configuration["fname"]
        fname = os.path.join(folder, fname)
        os.makedirs(os.path.dirname(fname), exist_ok=True)

        orbit = Orbit(self.configuration, parameters)
        t1 = orbit._backend.first_contact() - parameters["period"].to("day").value / 100
        t4 = (
            orbit._backend.fourth_contact() + parameters["period"].to("day").value / 100
        )
        t1 *= q.day.to(q.s)
        t4 *= q.day.to(q.s)
        delta_t = (t4 - t1) / nexposures
        nframes_per_exposure = int(delta_t / tframe)

        files = []
        with std_out_err_redirect_tqdm() as orig_stdout:
            for i in tqdm(range(nexposures), file=orig_stdout, dynamic_ncols=True):
                t0 = t1 + i * delta_t
                tso_transit = TSO(
                    ngrps=1, nints=nframes_per_exposure, star=star, target=name, t0=t0
                )

                # Set up transit parameters
                params = batman.TransitParams()
                # parameters["periastron"] # time of inferior conjunction
                params.t0 = 0
                # orbital period (days)
                params.per = parameters["period"].to_value("second")
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

                tso_transit.simulate(planet=planet, tmodel=tmodel, n_jobs=4)
                fname_i = fname.format(i)
                tso_transit.to_fits(fname_i)
                files.append(fname_i)

                # Fix observation date
                hdu = fits.open(fname_i)
                hdu[0].header["MJD-OBS"] = t0 * q.s.to(q.day)
                hdu[0].header["e_mu"] = orbit.get_mu(t0 * q.s.to(q.day))[()]
                hdu.writeto(fname_i, overwrite=True)

        return files

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

        # Subtract bias
        h_flat = fits.open(fname)
        d_flat = h_flat["SCI"].data
        f_bias = self.create_bias()
        h_bias = fits.open(f_bias)
        d_bias = h_bias["SCI"].data
        d_flat -= d_bias
        h_flat.writeto(fname, overwrite=True)
        try:
            os.remove(f_bias)
        except OSError:
            # File already removed in between
            pass

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
        flux_unit = q.erg / (q.Angstrom * q.cm ** 2 * q.s)
        planet = [np.copy(planet.wave), np.copy(planet.data)]
        star = [np.copy(star.wave) << q.Angstrom, np.copy(star.data) << flux_unit]
        star[0] = star[0].to(q.micrometer)
        star[1] *= STAR_DATA[1].max().value / star[1].max().value

        # TODO find a better way to create a planet spectrum from the transition spectrum
        planet[0] *= q.AA.to(q.micrometer)
        planet[1] = 1 - planet[1]
        planet[1] *= (PLANET_DATA[1].mean() - PLANET_DATA[1].min()) / planet[1].max()
        planet[1] += PLANET_DATA[1].min()

        self.create_wavelength()
        self.create_flat()
        self.create_observation(star, planet)

        folder = self.configuration["dir"]
        target = self.configuration["_star"] + self.configuration["_planet"]

        return "JWST_NIRISS", "GR700XD", target, folder

    def get_stellarflux(self, **data):
        """ Return the stellar flux for WASP107b """
        wave, flux = STAR_DATA
        wave = wave.to_value(q.AA)
        flux = flux.to_value(q.erg / (q.Angstrom * q.cm ** 2 * q.s))

        star = dataset(wave, flux)
        return star

    def get_planet(self, **data):
        """ Return the planet transmission for WASP107b """
        wave, flux = PLANET_DATA
        wave = wave * q.micrometer.to(q.AA)
        flux = flux - flux.min()
        flux /= flux.max()
        flux = 1 - flux

        planet = dataset(wave, flux)
        return planet
