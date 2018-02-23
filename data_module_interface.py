"""
Data Module provides an abstract class for all other data modules, e.g. harps, marcs
"""

from abc import ABC, abstractmethod
from scipy.interpolate import interp1d
from dataset import dataset


class data_module(ABC):
    """ An abstract class for data sources

    Contains various predefined abstract load functions.
    A class does not need to inherit all of these methods (in fact most won't),
    but rather they are a guideline for in- and output. Like a partial interface.
    """

    @classmethod
    @abstractmethod
    def load_observations(cls, conf, par, *args, **kwargs):
        """ load observations for this module

        Parameters:
        ----------
        conf : {dict}
            configuration settings
        par : {dict}
            stellar and planetary parameters

        *args

        **kwargs

        Raises
        ------
        NotImplementedError
            *abstract*
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load_tellurics(cls, conf, par, *args, **kwargs):
        """ Load the telluric transmission spectrum

        Parameters:
        ----------
        conf : {dict}
            configuration settings
        par : {dict}
            stellar and planetary parameters

        *args

        **kwargs

        Raises
        ------
        NotImplementedError
            *abstract*
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load_stellar_flux(cls, conf, par, *args, **kwargs):
        """ Load the stellar flux

        Parameters:
        ----------
        conf : {dict}
            configuration settings
        par : {dict}
            stellar and planetary parameters

        *args

        **kwargs

        Raises
        ------
        NotImplementedError
            *abstract*
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load_specific_intensities(cls, conf, par, *args, **kwargs):
        """ Load the stellar specific intensities

        Parameters:
        ----------
        conf : {dict}
            configuration settings
        par : {dict}
            stellar and planetary parameters

        *args

        **kwargs

        Raises
        ------
        NotImplementedError
            *abstract*
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def apply_modifiers(cls, conf, par, ds):
        """ apply modifiers for wavelength and flux from config file

        Parameters:
        ----------
        conf : {dict}
            configuration settings
        par : {dict}
            stellar and planetary parameters
        ds : {dataset}
            flux and wavelength to modify

        Raises
        ------
        NotImplementedError
            *abstract*
        """
        raise NotImplementedError

    @staticmethod
    def interpolate(newx, oldx, y, kind='linear'):
        """ interpolate between two grids

        Boundary errors are ignored and filled with 0s

        Parameters:
        ----------
        newx : {np.ndarray}
            new x values
        oldx : {np.ndarray}
            old x values
        y : {np.ndarray}
            old y values
        kind : {'linear', 'quadratic'}
            interpolation method (default is 'linear')
        Returns
        -------
        newy : np.ndarray
            interpolated values
        """

        return interp1d(oldx, y, bounds_error=False, fill_value=0, kind=kind)(newx)
