from idl_lib import *
from idl_lib import __array__
import _global


def myplanetcalc(my_value, srad, prad, distcentrum, np):
    """ calculate the MY values corresponding to different points on the planet """

    # MY=cos(asin(DistCent))
    # MY_value should be single value, not vector
    d = sin(acos(my_value))
    r = (dist / distcentrum / 2.) / srad

    phi = (dindgen(np)) / (np) * 2. * _global.pi
    myplanet = dblarr(np)

    x = sqrt(d**2. + r**2. - 2. * d * r * cos(phi))

    for n in range(np):
        if x[n] <= 1.:
            myplanet[n] = cos(asin(x[n]))

        if x[n] > 1.:
            myplanet[n] = 0.

    return myplanet
