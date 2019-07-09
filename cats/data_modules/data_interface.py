"""
Module for the abstract interfaces for various forms of data access
"""

from ..orbit import orbit as orbit_calculator

class data_interface:
    """ Abstract class for all data interfaces """
    # list of required steps in **data to work
    _requires = []

    def __init__(self, configuration):
        #:dict: Configuration for this module
        self.configuration = configuration

    def __str__(self):
        return self.__class__.__name__

class data_observations(data_interface):
    """
    Abstract class for Observation data modules
    Each module should provide access to all reduced data of the current star/planet
    """
    def get_observations(self, **data):
        """
        Return all data pertaining to the star/planet

        Parameters
        ----------
        **data : dict
            only contains orbital parameters

        Returns
        -------
        obs : dataset
            Observation data in a dataset container. All observations are given in one dateset within a single datacube.
            The observation time is given as parameter obs.time in MJD format.

        Raises
        ------
        NotImplementedError
            Abstract Class
        """
        raise NotImplementedError

class data_stellarflux(data_interface):
    """
    Abstract class for stellar flux data access
    """
    def get_stellarflux(self, **data):
        """
        Return the stellar flux for the current star
        Flux is normalized (?)

        Parameters
        ----------
        **data : dict
            previous data products

        Returns
        -------
        stellar_flux : dataset
            Stellar flux in a dataset container.

        Raises
        ------
        NotImplementedError
            Abstract class
        """
        raise NotImplementedError

class data_intensities(data_interface):
    def load_intensities(self, **data):
        # TODO: Cache results
        raise NotImplementedError

    def get_core(self, mu, intensity):
        # Step 2: Interpolate to planet position
        raise NotImplementedError

    def get_atmosphere(self, mu, intensity):
        raise NotImplementedError


    def get_intensities(self, **data):
        dates = data["observations"].time
        parameters = data["parameters"]
        intensity = self.load_intensities(**data)

        orbit = orbit_calculator(self.configuration, parameters)
        mu = orbit.get_mu(*orbit.get_pos(orbit.get_phase(dates)))

        i_core = self.get_core(mu, intensity)
        i_atmo = self.get_atmosphere(mu, intensity)
        return i_core, i_atmo

class data_tellurics(data_interface):
    def get_tellurics(self, **data):
        raise NotImplementedError

class data_orbitparameters(data_interface):
    def get_parameters(self, **data):
        # Guaranteed output fields (in SI units)
        # "period" in days
        # "periastron" in jd
        # "transit": in jd
        # "t_eff"
        # "logg" in log(cgs)
        # "monh"
        # "r_star"
        # "m_star"
        # "r_planet"
        # "m_planet"
        # "sma"
        # "inc" in degrees
        # "ecc"
        # "w" in degrees
        raise NotImplementedError

class data_planet(data_interface):
    def get_planet(self, **data):
        raise NotImplementedError

class data_raw(data_interface):
    def get_raw(self, **data):
        raise NotImplementedError

class data_reduction(data_interface):
    _requires = ["raw"]
    def get_reduced(self, **data):
        raise NotImplementedError
