
import numpy as np

def deltawavecreate(wl):
    deltawl = np.zeros_like(wl)

    deltawl[0] = wl[1] - wl[0]
    for n in range(1, len(wl)):
        deltawl[n] = wl[n] - wl[n - 1]

    # removing large jumps in deltawave due to jump in WLscale between spectral orders
    for n in range(2, len(wl)):
        if deltawl[n] > 3. * deltawl[n - 1]:
            deltawl[n] = deltawl[n - 1]

    dwl2 = 1. / deltawl**2.

    return dwl2
