from idl_lib import *
from idl_lib import __array__
import _global


def deltawavecreate(wl):

    deltawl = wl * 0.

    deltawl[0] = wl[1] - wl[0]
    for n in np.arange(1, n_elements(wl) + 1, 1):
        deltawl[n] = wl[n] - wl[n - 1]

    # removing large jumps in deltawave due to jump in WLscale between spectral orders
    for n in range(2, n_elements(wl) + 1):
        if deltawl(n) > 3. * deltawl(n - 1):
            deltawl[n] = deltawl[n - 1]

    dwl2 = 1. / deltawl**2.

    return dwl2
