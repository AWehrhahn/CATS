"""
A class that stores the original data in addition to any shifts etc, to conserve quality
"""
import numpy as np
import pandas as pd
from scipy.constants import c
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d

# Convert to km/s
c = c * 1e-3  # km/s

# TODO optimize


class dataset:

    def __init__(self, wl, flux, err=None, scale=1):
        self.__shift = None
        self.scale = scale
        self.__wl = wl
        self.__flux = flux
        if err is None:
            err = np.zeros_like(flux)
        self.__err = err
        self.__cache_flux__ = None
        self.__cache_err__ = None

    def __len__(self):
        return len(self.wl)

    def __getitem__(self, key):
        # Create a new object with changed values
        if self.flux.ndim == 2:
            _flux = self.flux[:, key]
            _err = self.err[:, key]
        else:
            _flux = self.flux[key]
            _err = self.err[key]

        return dataset(self.wl[key], _flux, _err)

    def __mul__(self, other):
        # if used in multiplication dataset acts as a proxy for the flux
        if isinstance(other, (float, int)):
            ds = dataset(self.wl, self.flux, self.err)
            ds.scale = other
            return ds
        if isinstance(other, np.ndarray):
            return dataset(self.wl, self.flux * other, self.err * other)
        if isinstance(other, dataset):
            if np.all(self.wl != other.wl):
                raise ValueError("Different wavelength scales")
            err = np.sqrt((other.flux * self.err)**2 +
                          (self.flux * other.err)**2)
            return dataset(self.wl, self.flux * other.flux, err)
        raise NotImplementedError

    def __truediv__(self, other):
        # if used in multiplication dataset acts as a proxy for the flux
        if isinstance(other, (float, int)):
            ds = dataset(self.wl, self.flux, self.err)
            ds.scale = 1 / other
            return ds
        if isinstance(other, np.ndarray):
            return dataset(self.wl, self.flux / other, self.err / other)
        if isinstance(other, dataset):
            if np.all(self.wl != other.wl):
                raise ValueError("Different wavelength scales")
            err = np.sqrt((self.err / other.flux)**2 +
                          (self.flux * other.err / other.flux**2)**2)
            return dataset(self.wl, self.flux / other.flux, err)
        raise NotImplementedError

    def __add__(self, other):
        if isinstance(other, (float, int)):
            return dataset(self.wl, self.flux + other, self.err)
        if isinstance(other, np.ndarray):
            return dataset(self.wl, self.flux + other, self.err)
        if isinstance(other, dataset):
            if np.all(self.wl != other.wl):
                raise ValueError("Different wavelength scales")
            err = np.sqrt(self.err**2 + other.err**2)
            return dataset(self.wl, self.flux + other.flux, err)
        raise NotImplementedError

    def __sub__(self, other):
        if isinstance(other, (float, int)):
            return dataset(self.wl, self.flux - other, self.err)
        if isinstance(other, np.ndarray):
            return dataset(self.wl, self.flux - other, self.err)
        if isinstance(other, dataset):
            if np.all(self.wl != other.wl):
                raise ValueError("Different wavelength scales")
            err = np.sqrt(self.err**2 + other.err**2)
            return dataset(self.wl, self.flux - other.flux, err)
        raise NotImplementedError

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return self * other

    def __radd__(self, other):
        return self * other

    def __rsub__(self, other):
        return self * other

    def __imul__(self, other):
        # if used in multiplication dataset acts as a proxy for the flux
        if isinstance(other, (float, int)):
            self.scale *= other
            return self
        if isinstance(other, np.ndarray):
            self.flux *= other
            self.err *= other
            return self
        if isinstance(other, dataset):
            if np.all(self.wl != other.wl):
                raise ValueError("Different wavelength scales")
            self.err = np.sqrt((other.flux * self.err)**2 +
                               (self.flux * other.err)**2)
            self.flux *= other.flux
            return self
        raise NotImplementedError

    def __itruediv__(self, other):
        # if used in multiplication dataset acts as a proxy for the flux
        if isinstance(other, (float, int)):
            self.scale /= other
            return self
        if isinstance(other, np.ndarray):
            self.flux /= other
            self.err /= other
            return self
        if isinstance(other, dataset):
            if np.all(self.wl != other.wl):
                raise ValueError("Different wavelength scales")
            self.err = np.sqrt((self.err / other.flux)**2 +
                               (self.flux * other.err / other.flux**2)**2)
            self.flux /= other.flux
            return self
        raise NotImplementedError

    def __iadd__(self, other):
        if isinstance(other, (float, int)):
            self.flux += other
            return self
        if isinstance(other, np.ndarray):
            self.flux += other
            return self
        if isinstance(other, dataset):
            if np.all(self.wl != other.wl):
                raise ValueError("Different wavelength scales")
            self.err = np.sqrt(self.err**2 + other.err**2)
            self.flux += other.flux
            return self
        raise NotImplementedError

    def __isub__(self, other):
        if isinstance(other, (float, int)):
            self.flux -= other
            return self
        if isinstance(other, np.ndarray):
            self.flux -= other
            return self
        if isinstance(other, dataset):
            if np.all(self.wl != other.wl):
                raise ValueError("Different wavelength scales")
            self.err = np.sqrt(self.err**2 + other.err**2)
            self.flux -= other.flux
            return self
        raise NotImplementedError

    def __neg__(self):
        return dataset(self.wl, -1 * self.flux, self.err)

    def __abs__(self):
        return dataset(self.wl, np.abs(self.flux), self.err)

    def interpolate(self, new, old, flux):
        if isinstance(flux, pd.DataFrame):
            values = interp1d(old, flux, kind='linear', bounds_error=False, fill_value=0, axis=0)(new)
            return pd.DataFrame(data=values, columns=flux.keys())
        return interp1d(old, flux, kind='linear', bounds_error=False, fill_value=0, axis=-1)(new)

    def doppler_shift(self, vel):
        #_c = c * 1e-3 #km/s
        self.wl = (1 + vel / c) * self.wl  # shifted wavelength range

    def gaussbroad(self, sigma):
        self.flux = gaussian_filter1d(self.flux, sigma)
        self.err = gaussian_filter1d(self.err, sigma) #TODO???

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
                self.__cache_flux__ = self.interpolate(
                    self.__shift, self.__wl, self.__flux) * self.scale
            return self.__cache_flux__

    @flux.setter
    def flux(self, value):
        # interpolate to old wavelength grid?
        self.__cache_flux__ = None
        if self.__shift is None:
            if value.shape[-1] != self.wl.shape[0]:
                raise ValueError(
                    "Size doesn't match, consider only changing the wavelenght")
            self.__flux = value
        else:
            # make new wavelength grid permanent
            self.__flux = value
            self.__err = self.err
            self.__wl = self.__shift
            self.__shift = None

    @property
    def err(self):
        if self.__shift is None:
            return self.__err * self.scale
        else:
            if self.__cache_err__ is None:
                self.__cache_err__ = self.interpolate(
                    self.__shift, self.__wl, self.__err) * self.scale
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
