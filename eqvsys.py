from idl_lib import trisol
import numpy as np


def eqvsys(f, g, wl, dwl2, _lambda):

    lam = _lambda * dwl2

    a = np.zeros_like(f[0, :])
    a = a - lam
    c = a

    b = np.sum(g**2., 2)
    b = b + (2. * lam)
    r = np.sum((f * g), 2)

    b[0] = b[0] - lam[0]
    b[len(a)] = b[len(a)] - lam[len(a)]

    return trisol(a, b, c, r)
