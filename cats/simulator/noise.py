"""
Put different noise statistics in here
"""

import numpy as np


class NoiseBase:
    pass


class WhiteNoise(NoiseBase):
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, size, data=None):
        noise = np.zeros(size)
        if hasattr(self.sigma, "__iter__"):
            for i, s in enumerate(self.sigma):
                noise[i] = np.random.normal(scale=s, size=size[1])
        else:
            # sigma is a scalar ?
            noise = np.random.normal(scale=self.sigma, size=size)
        return noise


class WhiteNoisePercentage(NoiseBase):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def __call__(self, size, data):
        noise = np.zeros(size)
        if hasattr(self.sigma, "__iter__"):
            for i, s in enumerate(self.sigma):
                noise[i] = np.random.normal(scale=s, size=size[1])
        else:
            # sigma is a scalar ?
            noise = np.random.normal(scale=self.sigma, size=size)

        noise = data * noise

        return noise


class BadPixelNoise(NoiseBase):
    """
    Additional noise due to bad pixels in the detector
    This adds additional noise to the spectrum, in random places
    
    Parameters
    ----------
    size : int
        length of the spectrum
    bad_pixels_per_element : float
        expected bad pixels per resolution element
    sigma : float
        width of the additional noise introduces by 1 bad pixel
    
    Returns
    -------
    noise : array
        bad pixel noise
    """

    def __init__(self, bad_pixels_per_element, sigma):
        self.bad_pixels_per_element = bad_pixels_per_element
        self.sigma = sigma

    def __call__(self, size, data=None):
        nsize = np.product(size)
        number_bad_pixels = int(self.bad_pixels_per_element * nsize)
        bad_pixels = np.random.choice(nsize, size=number_bad_pixels)

        noise = np.zeros(size)
        noise.ravel()[bad_pixels] += np.random.normal(
            scale=self.sigma, size=number_bad_pixels
        )

        return noise


class PoisonNoise(NoiseBase):
    def __init__(self, scaling):
        super().__init__()
        self.scaling = scaling

    def __call__(self, size, data):
        sigma = [np.sqrt(d.decompose()) for d in data]
        noise = np.zeros(size)
        for i, s in enumerate(sigma):
            s = np.nan_to_num(s)
            noise[i] = np.random.poisson(lam=s)
        noise *= self.scaling
        return noise
