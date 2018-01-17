"""
A class that stores the original data in addition to any shifts etc, to conserve quality
"""

from scipy.interpolate import interp1d

#TODO is this better than what I have?
#TODO if yes implement this

class dataset:

    def __init__(self, wl, flux):
        self.__shift = None
        self.scale = 1
        self.__wl = wl
        self.__flux = flux

    def interpolate(self, new, old, flux):
        return interp1d(old, flux, kind='quadratic', bounds_error=False)(new)

    @property
    def wl(self):
        if self.__shift is None:
            return self.__wl
        else:
            return self.__shift

    @wl.setter
    def _set_wl(self, value):
        self.__shift = value

    @property
    def flux(self):
        if self.__shift is None:
            return self.__flux * self.scale
        else:
            return self.interpolate(self.__shift, self.__wl, self.__flux) * self.scale

    @flux.setter
    def _set_flux(self, value):
        # interpolate to old wavelength grid?
        if self.__shift is None:
            self.__flux = value
        else:
            self.__flux = self.interpolate(self.__wl, self.__shift, value)
