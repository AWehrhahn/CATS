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

class dataset:
    """
    Simpler version of dataset, that does not cache flux/err but just performs the calculations required
    """

    def __init__(self, wl, flux, err=None):
        self.__wl = wl
        if flux.ndim == 1:
            flux = flux[None, :]
        self.__flux = flux
        if err is not None:
            if err.ndim == 1:
                err = err[None, :]
            self.__err = err
        else:
            self.__err = np.zeros_like(flux)
            if isinstance(flux, pd.DataFrame):
                self.__err = self.__err.swapaxes(0, 1)

    def __interpolate__(self, new, old, flux):
        # Avoid unnecessary interpolation
        if len(new) == len(old) and np.all(new == old):
            return flux

        # if flux is a DataFrame the axes are swapped
        isDataFrame = isinstance(flux, pd.DataFrame)
        if isDataFrame:
            keys = flux.keys()
            flux = flux.values.swapaxes(0, 1)

        kind = "linear"
        fill_value = np.nan
        mask = ~np.isnan(flux)
        ndim = flux.shape[0]
        if old.ndim > 1:
            ndim = old.shape[0]

        res = np.full((ndim, new.shape[0]), np.nan, dtype=float)
        for i in range(ndim):
            # fix dimensions
            _old = old[i] if old.ndim > 1 else old
            _flux = flux[i] if flux.shape[0] > 1 else flux[0]
            _mask = mask[i] if mask.shape[0] > 1 and ndim == 1 else mask[0]

            if len(np.where(mask)[0]) > 0: 
                res[i] = interp1d(_old[_mask], _flux[_mask], kind=kind,
                              bounds_error=False, fill_value=fill_value)(new)
            #res[i] = np.clip(res[i], 0, 1)

        if isDataFrame:
            res = res.swapaxes(0, 1)
            return pd.DataFrame(data=res, columns=keys)
        return res

    def __len__(self):
        return len(self.wl)

    def __getitem__(self, key):
        # Create a new object with changed values
        if self.flux.ndim == 2:
            _flux = self.flux[:, key]
            _err = self.err[:, key]
            _wl = self.wl[key]
        else:
            _flux = self.flux[key]
            _err = self.err[key]
            _wl = self.wl[key]

        return dataset(_wl, _flux, _err)

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

    def doppler_shift(self, vel):
        """ doppler shift spectrum """
        if isinstance(vel, np.ndarray) and vel.ndim == 1:
            vel = vel[:, None]

        shift = self.wl * (1 + vel / c)
        self.flux = self.__interpolate__(self.wl, shift, self.flux)
        self.err = self.__interpolate__(self.wl, shift, self.err)

    def gaussbroad(self, sigma):
        """ broaden spectrum """
        self.flux = gaussian_filter1d(self.flux, sigma)

    def write(self, fname):
        """ save dataset to disk """
        data = np.array([self.wl, self.flux, self.err]).swapaxes(0, 1)
        np.savetxt(fname, data, delimiter=', ')

    def change_grid(self, value):
        """ actually change the underlying wavelength grid and not the shifted wavelength

        Parameters:
        ----------
        value : {np.ndarray}
            new wavelength grid
        """
        self.__wl = value

    @property
    def wl(self):
        """ Wavelength """
        return self.__wl

    @wl.setter
    def wl(self, value):
        self.__flux = self.__interpolate__(value, self.__wl, self.__flux)
        self.__err = self.__interpolate__(value, self.__wl, self.__err)
        self.__wl = value

    @property
    def flux(self):
        """ Flux """
        return self.__flux

    @flux.setter
    def flux(self, value):
        if value.ndim == 1:
            value = value[None, :]
        #assert self.flux.shape == value.shape
        self.__flux = value

    @property
    def err(self):
        """ Error on flux """
        return self.__err

    @err.setter
    def err(self, value):
        if value.ndim == 1:
            value = value[None, :]
        #assert self.err.shape == value.shape
        self.__err = value

    @property
    def scale(self):
        """ Scale of the flux """
        return 1

    @scale.setter
    def scale(self, value):
        self.flux *= value
