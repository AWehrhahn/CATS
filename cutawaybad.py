from idl_lib import where
import numpy as np
from radialvelocity import radialvelocity
from planetvelocityco import planetvelocityco


def cutawaybad(wl, f, g, par):
    """
    Remove "bad" parts of the wavelength spectrum
    Only on the outside edges
    """
    # this is for simulated HD209458b
    bvelocities = radialvelocity(par)
    pvelocities = planetvelocityco(par)

    min_vel = min(-(bvelocities + pvelocities))
    max_vel = max(-(bvelocities + pvelocities))

    #Speed of Light in km/s
    c = 299792.
    wlmax = max(wl) * (np.sqrt((1. + min_vel / c)/(1.-min_vel/c)))
    wlmin = min(wl) * (np.sqrt((1. + max_vel / c)/(1.-max_vel/c)))

    index = where(wl > wlmin and wl < wlmax)
    wl = wl[index]
    f = f[:, index]
    g = g[:, index]

    return wl, f, g
