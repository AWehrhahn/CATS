from idl_lib import trisol
import numpy as np


def eqvsys(f, g, wl, dwl2, _lambda):

    lam = _lambda * dwl2

    a = c = -lam 

    b = np.sum(g**2., 1) + (2. * lam)
    r = np.sum((f * g), 1)

    # Compensate for having only two summands in the first and last row
    b[0] = b[0] - lam[0]
    b[-1] = b[-1] - lam[-1]

    return trisol(a, b, c, r)
