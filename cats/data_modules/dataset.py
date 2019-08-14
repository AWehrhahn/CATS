import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d
from scipy.constants import speed_of_light

# speed of light in km/s
c = speed_of_light * 1e-3


class dataset:
    """ Contains the original data (and errors) on a wavelength scale, and offers functionality to shift it as desired """

    def __init__(self, wave, data, err=None):
        self._wave_orig = wave
        self._data_orig = data
        self._err_orig = err
        self.time = 0

        self.broadening = 0
        self.scale = 1

    @property
    def wave(self):
        return self._wave_orig

    @property
    def data(self):
        result = self._data_orig
        result = self._broaden(result)
        result *= self.scale

        return result
    
    @property
    def error(self):
        if self._err_orig is None:
            result = np.ones_like(self._data_orig)
        else:
            result = self._err_orig
        result = self._broaden(result)
        result *= self.scale

        return result

    def _interpolate(self, data, old_wave, new_wave):
        # mask = (new_wave > old_wave.min()) & (new_wave < old_wave.max())
        # flux = np.zeros(len(new_wave))
        # flux[mask] = spectres.spectres(new_wave[mask], old_wave, data)
        flux = interp1d(old_wave, data, bounds_error=False, fill_value=0)(new_wave)
        return flux, 1

    def _broaden(self, data):
        if self.broadening == 0:
            return data
        else:
            return gaussian_filter1d(data, self.broadening)

    def __len__(self):
        return len(self.wave)

    def __add__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            return dataset(self.wave, self.data + other, self.error)
        if isinstance(other, dataset):
            shifted = other._interpolate(other.data, other.wave, self.wave)
            data = self.data + shifted[0]
            error = np.sqrt(self.error ** 2 + shifted[1] ** 2)
            return dataset(self.wave, data, error)
        raise NotImplementedError

    def __sub__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            return dataset(self.wave, self.data - other, self.error)
        if isinstance(other, dataset):
            shifted = other._interpolate(other.data, other.wave, self.wave)            
            data = self.data - shifted[0]
            error = np.sqrt(self.error ** 2 + shifted[1] ** 2)
            return dataset(self.wave, data, error)
        raise NotImplementedError

    def __mul__(self, other):
        if isinstance(other, (int, float, np.ndarray)):
            return dataset(self.wave, self.data * other, self.error)
        if isinstance(other, dataset):
            shifted = other._interpolate(other.data, other.wave, self.wave)            
            data = self.data * shifted[0]
            error = np.sqrt((self.error * shifted[0])**2 + (shifted[1] * self.data)**2)
            return dataset(self.wave, data, error)
        raise NotImplementedError

    def shift(self, rv, i=None):
        if i is None:
            i = slice(None, None, None)
        new_wave = self._wave_orig * (1 + rv/c)
        new_flux, new_error = self._interpolate(self._data_orig[i], self.wave, new_wave)

        self._wave_orig = new_wave
        self._data_orig[i] = new_flux
        if self._err_orig is not None:
            self._err_orig[i] = new_error

    def new_grid(self, new_wave):
        new_flux, new_error = self._interpolate(self._data_orig, self.wave, new_wave)
        self._wave_orig = new_wave
        self._data_orig = new_flux        
        if self._err_orig is not None:
            self._err_orig = new_error
        