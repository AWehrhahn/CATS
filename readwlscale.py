import numpy as np


def readwlscale(path, wl_file):
    filename = path + 'wavelength/' + wl_file + '.dat'
    print(wl_file)
    return np.loadtxt(filename, ndmin=2)
