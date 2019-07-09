from itertools import product

from .data_interface import data_raw

# TODO get data for other stars/planets
from awesimsoss import TSO, PLANET_DATA, STAR_DATA
import batman
from astropy import units as q

class awesimsoss(data_raw):
    def get_raw(self, **data):
        #TODO get proper filename
        fname = "my_SOSS_simulation.fits"
        # Set up simulation object
        ngrps = self.configuration["ngroups"]
        nints = self.configuration["nintegrations"]
        tso_transit = TSO(ngrps=ngrps, nints=nints, star=STAR_DATA)

        # Set up transit parameters
        parameters = data["parameters"]

        params = batman.TransitParams()
        params.t0 = 0 #parameters["periastron"] # time of inferior conjunction
        params.per = parameters["period"] # orbital period (days)
        params.a = parameters["sma"] * q.km.to(q.R_sun) #0.0558*q.AU.to(q.R_sun)*0.66 # semi-major axis (in units of stellar radii)
        params.rp = parameters["r_planet"] / parameters["r_star"] # radius ratio for Jupiter orbiting the Sun
        params.inc = parameters["inc"] # orbital inclination (in degrees)
        params.ecc = parameters["ecc"] # eccentricity
        params.w = parameters["w"] # longitude of periastron (in degrees) p
        
        params.limb_dark = 'quadratic' # limb darkening profile to use
        params.u = [0.1,0.1] # limb darkening coefficients

        tmodel = batman.TransitModel(params, tso_transit.time)
        tmodel.teff = parameters["teff"] # effective temperature of the host star
        tmodel.logg = parameters["logg"] # log surface gravity of the host star
        tmodel.feh = parameters["monh"] # metallicity of the host star

        tso_transit.simulate(planet=PLANET_DATA, tmodel=tmodel)
        tso_transit.to_fits(fname)

        # TODO Reduce raw images

        return fname
