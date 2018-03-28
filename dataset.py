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

# TODO decide which version of dataset is better


class dataset_old:
    """
    Stores the wavelength, the flux, and the errors of a single spectrum and provides some simple functionality
    Also supports math operators together with scalars, arrays, and other datasets

    It tries to conserve the original data seperate from its current representation for as long as possible, so as to avoid data loss
    Therefore any changes to the wavelength are stored seperately and the flux is interpolated only when necessary from the original grid

    Properties:
    ----------
    wl : np.ndarray
        Wavelength
    flux : np.ndarray
        Flux
    err : np.ndarray
        Errors on the flux
    scale : float
        Multiplier for the flux and error
    gaussian : float
        Gaussian broadening sigma

    Functions:
    ----------
    __init__(wl, flux, err=None, scale=1, gaussian=None)

    doppler_shift(velocity)
        doppler shift flux and wavelength

    gaussbroad(sigma)
        apply gaussian broadening to the flux

    change_grid(value)
        change the underlying wavelength grid
    """

    def __init__(self, wl, flux, err=None, scale=1, gaussian=None):
        self.scale = scale
        self.gaussian = gaussian
        self.__wl = wl
        self.__shift = wl
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

    def __interpolate__(self, new, old, flux):
        # Avoid unnecessary interpolation
        if len(new) == len(old) and np.all(new == old):
            return flux

        if isinstance(flux, pd.DataFrame):
            values = interp1d(old, flux, kind='zero',
                              bounds_error=False, fill_value=0, axis=0)(new)
            return pd.DataFrame(data=values, columns=flux.keys())
        return interp1d(old, flux, kind='zero', bounds_error=False, fill_value=0, axis=-1)(new)

    def doppler_shift(self, vel):
        """ Doppler shift the flux spectrum and the wavelength

        Parameters:
        ----------
        vel : {float, array}
            velocity in km/s to doppler shift. Negative is moving towards Earth, positive away
        """
        if isinstance(vel, np.ndarray) and vel.ndim == 1:
            vel = vel[:, None]
            #self.__wl = self.wl[None, :]
        self.__wl = (1 + vel / c) * self.wl
        self.wl = self.__wl

    def gaussbroad(self, sigma):
        """ Apply gaussian broadening to the flux

        Parameters:
        ----------
        sigma : {float}
            width of the kernel in pixel
        """
        if sigma is None:
            self.gaussian = None
        elif self.gaussian is None:
            self.gaussian = sigma
        else:
            self.gaussian += sigma

    def __gaussbroad_actual__(self, flux):
        if self.gaussian is not None:
            return gaussian_filter1d(flux, self.gaussian)
        return flux

    def write(self, fname):
        data = np.array([self.wl, self.flux, self.err]).swapaxes(0, 1)
        np.savetxt(fname, data, delimiter=', ')

    @property
    def wl(self):
        """ The wavelength array of the spectrum """
        return self.__shift

    @wl.setter
    def wl(self, value):
        self.__shift = np.copy(value)
        self.__cache_flux__ = None
        self.__cache_err__ = None

    def change_grid(self, value):
        """ actually change the underlying wavelength grid and not the shifted wavelength

        Parameters:
        ----------
        value : {np.ndarray}
            new wavelength grid
        """

        self.__wl = value
        self.__shift = value

        self.__cache_flux__ = None
        self.__cache_err__ = None

    @property
    def flux(self):
        """ The flux values of the spectrum, may be 2dimensional for several observations """
        if self.__cache_flux__ is None:
            self.__cache_flux__ = self.__interpolate__(
                self.__shift, self.__wl, self.__flux) * self.scale
            self.__cache_flux__ = self.__gaussbroad_actual__(
                self.__cache_flux__)
        return self.__cache_flux__

    @flux.setter
    def flux(self, value):
        # Check if the size matches
        if value.shape[-1] != self.wl.shape[0]:
            raise ValueError(
                "Size doesn't match, consider only changing the wavelenght")

        self.__flux = np.copy(value)
        self.scale = 1
        # if wavelength grid stayed the same
        if self.__shift is not self.__wl:
            # make new wavelength grid permanent
            self.__err = self.err
            self.__wl = self.__shift

        # Invalidate Cache
        self.__cache_flux__ = None
        self.__cache_err__ = None

    @property
    def err(self):
        """ The absolute errors on the flux """
        if self.__cache_err__ is None:
            self.__cache_err__ = self.__interpolate__(
                self.__shift, self.__wl, self.__err) * self.scale
        return self.__cache_err__

    @err.setter
    def err(self, value):
        self.__cache_err__ = None
        self.__err = value / self.scale
        if self.__shift is not self.__wl:
            self.__flux = self.flux
            self.__wl = self.__shift


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
        assert self.err.shape == value.shape
        self.__err = value

    @property
    def scale(self):
        """ Scale of the flux """
        return 1

    @scale.setter
    def scale(self, value):
        self.flux *= value
