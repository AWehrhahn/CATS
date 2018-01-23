"""
Data Module provides an abstract class for all other data modules, e.g. harps, marcs
"""

from abc import ABC, abstractmethod
from scipy.interpolate import interp1d

class data_module(ABC):
    """
    An abstract class that defines all the different load routines to get data
    A class does not need to inherit all of these methods (in fact most won't),
    but rather they are a guideline for in- and output
    """

    @classmethod
    @abstractmethod
    def load_observations(cls, conf:dict, par:dict, *args, **kwargs)->(list, list):
        """ Load transit observations """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load_tellurics(cls, conf:dict, par:dict, *args, **kwargs)->(list, list):
        """ Load the telluric spectrum """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load_stellar_flux(cls, conf:dict, par:dict, *args, **kwargs)->(list, list):
        """ Load stellar flux """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load_specific_intensities(cls, conf:dict, par:dict, *args, **kwargs)->(list, (list, list)):
        """ Load specific intensities """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def apply_modifiers(cls, conf:dict, par:dict, wl:list, flux:list)->(list, list):
        """ apply modifiers for wavelength and flux from config file """
        raise NotImplementedError

    @staticmethod
    def interpolate(newx, oldx, y, kind='linear'):
        """ interpolate without errors """
        return interp1d(oldx, y, bounds_error=False, fill_value=0)(newx)
