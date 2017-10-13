from idl_lib import *
from idl_lib import __array__
import _global


def radialvelocity(radialvelstart, radialvelend, nexposures):
    """ linear function from start velocity to end velocity """
    velocities = (dindgen(nexposures)) / (nexposures - 1.) * \
        (radialvelend - radialvelstart) + radialvelstart

    return velocities
