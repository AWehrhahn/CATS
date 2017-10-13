from idl_lib import *
from idl_lib import __array__
import _global


def intensityinterpolation(my_value, intspecall):
    """ IntspecAll should have index equal to MYvalue, so index 0 should be MY=0, where all values are 0 """

    myi = my_value * 10.
    i1 = floor(myi)  # index 1
    i2 = ceil(myi)  # index 2
    diff = myi - floor(myi)

    fac1 = 1. - diff  # multiplicator factor for i index 1
    fac2 = diff

    return intspecall[i1, :] * fac1 + intspecall[i2, :] * fac2
