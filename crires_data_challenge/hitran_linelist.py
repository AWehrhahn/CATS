import pandas as pd
import numpy as np
import astropy.units as u

from os.path import dirname, join


class Hitran:
    def __init__(self):
        fname = join(dirname(__file__), "../data/h2o.par")

        df = pd.read_table(fname)
        self.table = df
        self.wavelength = 2 * np.pi / self.table["nu"]
        self.wavelength = self.wavelength << u.Unit("cm")
        self.wavelength = self.wavelength.to("AA")
