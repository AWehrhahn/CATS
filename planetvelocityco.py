from idl_lib import dindgen
from numpy import pi, arctan as atan, sin


def planetvelocityco(par):
    i = par.inclination * pi / 180.  # radians

    veloorbit = par.semimajoraxis * 2. * pi / \
        par.period  # planet velocity in stellar restframe

    # each exposure's time from peri
    sec_exps = (dindgen(par.nexposures) /
                (par.nexposures - 1) - 0.5) * par.transitduration

    # fraction of full orbit from peri that each exp is taken
    tranitfraction = sec_exps / par.period
    # converted fraction to angle in radians
    anglesexp = tranitfraction * 2. * pi

    # distances as seen from observer edge on, to centre of star
    distexp = atan(anglesexp) * par.semimajoraxis
    pvelocities = veloorbit * distexp / par.semimajoraxis
    # velocities as seen from us, with inclination
    pvelocities = pvelocities * sin(i)
    pvelocities = abs(pvelocities)

    for ex in range(par.nexposures // 2, par.nexposures):
        pvelocities[ex] = pvelocities[ex] * (-1.)

    return pvelocities
