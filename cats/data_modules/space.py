import numpy as np
from .dataset import dataset
from .data_interface import data_tellurics

class space(data_tellurics):
    """
    Tellurics for space based telescopes,
    i.e. No telluric absorption features,
    will just return 1
    """

    def get_tellurics(self, **data):
        wave = data["observations"].wave
        flux = np.ones(wave.size)
        ds = dataset(wave, flux)
        return ds
