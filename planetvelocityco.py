from idl_lib import *
from idl_lib import __array__
import _global


def planetvelocityco(inclination, nexposures, semimajoraxis, period, srad, prad, transitduration):
    i = inclination * _global.pi / 180.  # radians

    veloorbit = semimajoraxis * 2. * _global.pi / \
        period  # planet velocity in stellar restframe

    # each exposure's time from peri
    sec_exps = (dindgen(nexposures) / (nexposures - 1) - 0.5) * transitduration

    # fraction of full orbit from peri that each exp is taken
    tranitfraction = sec_exps / period
    # converted fraction to angle in radians
    anglesexp = tranitfraction * 2. * _global.pi

    # distances as seen from observer edge on, to centre of star
    distexp = atan(anglesexp) * semimajoraxis
    pvelocities = veloorbit * distexp / semimajoraxis
    # velocities as seen from us, with inclination
    pvelocities = pvelocities * sin(i)
    pvelocities = abs(pvelocities)

    for ex in np.arange(nexposures / 2, nexposures - 1 + 1, 1):
        pvelocities[ex] = pvelocities[ex] * (-1.)

    return pvelocities
