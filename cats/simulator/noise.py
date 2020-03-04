"""
Put different noise statistics in here
"""

import numpy as np

class NoiseBase:
    pass

class WhiteNoise(NoiseBase):
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, size):
        return np.random.normal(scale=self.sigma, size=size)

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

    def __call__(self, size):
        nsize = np.product(size)
        number_bad_pixels = int(self.bad_pixels_per_element * nsize)
        bad_pixels = np.random.choice(nsize, size=number_bad_pixels)

        noise = np.zeros(size)
        noise.ravel()[bad_pixels] += np.random.normal(scale=self.sigma, size=number_bad_pixels)

        return noise
