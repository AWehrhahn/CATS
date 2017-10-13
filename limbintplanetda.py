from idl_lib import *
from idl_lib import __array__
import _global

from myatmocalc import myatmocalc
from intensityinterpolation import intensityinterpolation


def limbintplanetda(my_value, intspecall, srad, prad, atmoheight, wl):
    # number of points in the atmosphere
    np = 20.  # number of angle points
    mr = 20.  # number of radial distances

    int = dblarr(n_elements(wl), mr)
    intplanet = dblarr(n_elements(wl))
    for d in np.arange(0., mr - 1 + 1, 1):
        distcentrum = (d + 0.5) / (mr) * (prad + atmoheight)
        myplanet = myatmocalc(
            my_value, srad, (prad + atmoheight), distcentrum, np)
        # create mean intensity spectra for all atmosphere points
        for n in range(np):
            int[d, :] = int[d, :] + \
                intensityinterpolation(myplanet[n], intspecall)

        int[d, :] = int[d, :] / np

        # factor in area of circle
        areafac = (distcentrum / (prad + atmoheight) + ((0.5) / mr)) ** 2. - \
            (distcentrum / (prad + atmoheight) - ((0.5) / mr)) ** 2.
        int[d, :] = areafac * int[d, :]

    for d in range(mr):
        intplanet = intplanet + int[d, :]

    return intplanet
