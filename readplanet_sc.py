import numpy as np
from wlinterpolateco import wlinterpolateco


def readplanet_sc(wl, files):

    filename = files.path + 'exoplanet/' + files.exoplanet + '.dat'

    indata = np.loadtxt(filename, ndmin=2)

    planetspect = abs(np.transpose(indata[:, 1]))
    wlt = abs(np.transpose(indata[:, 0]))
    return wlinterpolateco(planetspect, wlt, wl)
