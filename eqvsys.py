from idl_lib import *
from idl_lib import __array__
import _global


def eqvsys(f, g, wl, dwl2, _lambda):

    lam = _lambda * dwl2
    # lam=lambda

    a = f[0, :] * 0.
    a = a - lam
    c = a

    b = total(g**2., 2)
    b = b + (2. * lam)
    r = total((f * g), 2)

    b[0] = b[0] - lam[0]
    # b(n_elements(a)-1) = b(n_elements(a)-1) - lam(n_elements(a)-1)
    b[n_elements(a)] = b[n_elements(a)] - lam[n_elements(a)]

    solution = trisol(a, b, c, r)

    return solution
