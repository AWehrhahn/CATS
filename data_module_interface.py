"""
Data Module provides an abstract class for all other data modules, e.g. harps, marcs
"""

from abc import ABC, abstractmethod

class data_module(ABC):
    """
    An abstract class that defines all the different load routines to get data
    """

    @classmethod
    @abstractmethod
    def load_observations(cls, conf, par, *args, **kwargs):
        """ Load transit observations """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load_tellurics(cls, conf, par, *args, **kwargs):
        """ Load the telluric spectrum """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load_stellar_flux(cls, conf, par, *args, **kwargs):
        """ Load stellar flux """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load_specific_intensities(cls, conf, par, *args, **kwargs):
        """ Load specific intensities """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def apply_modifiers(cls, conf, par, wl, flux):
        """ apply modifiers for wavelength and flux from config file """
        raise NotImplementedError