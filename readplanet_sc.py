from idl_lib import *
from idl_lib import __array__
import _global

from wlinterpolateco import wlinterpolateco


def readplanet_sc(pathname, exoplanetfilename, wl):

    filename = pathname + 'exoplanet/' + exoplanetfilename + '.dat'

    filelength = 194187.
    indata = dblarr(2, filelength)
    openr(1, filename)
    indata = readf(1)
    close(1)

    # PlanetSpec=WLinterpolateCO(indata(1,*),indata(0,*),WL)
    planetspect = abs(transpose(indata[:, 1]))
    wlt = abs(transpose(indata[:, 0]))
    planetspec = wlinterpolateco(planetspect, wlt, wl)
    return planetspec
