"""
A class that stores the original data in addition to any shifts etc, to conserve quality
"""
from scipy.constants import c
from scipy.interpolate import interp1d


c = c * 1e-3 #km/s
#TODO is this better than what I have?
#TODO if yes implement this

class dataset:

    def __init__(self, wl, flux, err):
        self.__shift = None
        self.scale = 1
        self.__wl = wl
        self.__flux = flux
        self.__err = err
        self.__cache_flux__ = None
        self.__cache_err__ = None

    def __getitem__(self, key):
        #Create a new object with changed values
        return dataset(self.wl[key], self.flux[key], self.err[key])

    def interpolate(self, new, old, flux):
        return interp1d(old, flux, kind='quadratic', bounds_error=False)(new)

    def doppler_shift(self, vel):
        #_c = c * 1e-3 #km/s
        self.wl = (1 + vel / c) * self.wl  # shifted wavelength range

    @property
    def wl(self):
        if self.__shift is None:
            return self.__wl
        else:
            return self.__shift

    @wl.setter
    def wl(self, value):
        self.__shift = value
        self.__cache_flux__ = None
        self.__cache_err__ = None

    @property
    def flux(self):
        if self.__shift is None:
            return self.__flux * self.scale
        else:
            if self.__cache_flux__ is None:
                self.__cache_flux__ = self.interpolate(self.__shift, self.__wl, self.__flux) * self.scale
            return self.__cache_flux__

    @flux.setter
    def flux(self, value):
        # interpolate to old wavelength grid?
        self.__cache_flux__ = None
        if self.__shift is None:
            self.__flux = value
        else:
            # make new wavelength grid permanent
            self.__flux = value
            self.__err = self.err
            self.__wl = self.__shift
            self.__shift = None
            
            #self.__flux = self.interpolate(self.__wl, self.__shift, value)

    @property
    def err(self):
        if self.__shift is None:
            return self.__err * self.scale
        else:
            if self.__cache_err__ is None:
                self.__cache_err__ = self.interpolate(self.__shift, self.__wl, self.__err) * self.scale
            return self.__cache_err__

    @err.setter
    def err(self, value):
        self.__cache_err__ = None
        if self.__shift is None:
            self.__err = value
        else:
            self.__flux = self.flux
            self.__err = value
            self.__wl = self.__shift
            self.__shift = None
