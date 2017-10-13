from idl_lib import *
from idl_lib import __array__
import _global

from numpy import pi


def myatmocalc(my_value, srad, prad, atmoheight, np):
    """ calculate the MY values corresponding to different points in the atmosphere """

    # MY=cos(asin(DistCent))
    # MY_value should be single value, not vector
    d = sin(acos(my_value))
    r = (prad + atmoheight / 2.) / srad

    phi = (dindgen(np)) / (np) * 2. * pi
    myatmo = dblarr(np)

    x = sqrt(d**2. + r**2. - 2. * d * r * cos(phi))

    for n in range(np):
        if x(n) <= 1.:
            myatmo[n] = cos(asin(x[n]))

        if x(n) > 1.:
            myatmo[n] = 0.

    return myatmo
