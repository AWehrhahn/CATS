from idl_lib import *
from idl_lib import __array__
import _global

from radialvelocity import radialvelocity
from planetvelocityco import planetvelocityco


def cutawaybad(nexp, wl, f, g, radialvelstart, radialvelend, semimajoraxis, period, srad, prad, inclination, transitduration):

    # this is for simulated HD209458b
    bvelocities = radialvelocity(radialvelstart, radialvelend, nexp)
    pvelocities = planetvelocityco(
        inclination, nexp, semimajoraxis, period, srad, prad, transitduration)

    min_vel = min(-(bvelocities + pvelocities))
    max_vel = max(-(bvelocities + pvelocities))

    wlmax = max(wl) * (sqrt((1. + min_vel / 299792.)))
    wlmin = min(wl) * (sqrt((1. + max_vel / 299792.)))

    index = where(wl > wlmin and wl < wlmax)
    wl = wl[index]
    f = f[:, index]
    g = g[:, index]

    return wl
