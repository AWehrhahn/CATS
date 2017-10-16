from idl_lib import *
from idl_lib import __array__
import _global

from numpy import pi, sin, cos
from numpy import arcsin as asin  # don't lie to me pylint, that totally exists


def myvect(par):
    i = par.inclination * pi / 180.
    d = sin(pi / 2. - i) * par.semimajoraxis
    totdist = 2. * sqrt(par.srad**2. - d**2.)

    # how much of planet should be within stellar disk before we start using the exposures, in km
    entpla = par.prad
    # for 0 we start when planet centrum touches stellar disk edge

    distances = (dindgen(par.nexposures) / (par.nexposures - 1.)
                 * 2. * (totdist / 2. - entpla)) / par.srad
    distances = distances - total(distances) / n_elements(distances)

    # distance from centre of stellar disk to centre of planet
    distcent = sqrt(abs(d / par.srad))

    my = cos(asin(distcent))

    return my
