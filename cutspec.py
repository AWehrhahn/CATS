from idl_lib import *
from idl_lib import __array__
import _global


def cutspec(inspec, wl):

    startp = wl[0]
    endp = wl[n_elements(wl)]

    # starting condiction
    s = 0.
    while inspec(s, 0) < startp:
        s = s + 1.

    # if s eq 0 do begin
    # s=2
    # endif

    startpoint = s - 1.

    # ending condition
    e = 1.
    while inspec(e, 0) < endp:
        e = e + 1.
    endpoint = e

    outspec = dblarr(endpoint - startpoint + 1, 2)
    outspec[0, :] = inspec[0, startpoint:endpoint + 1]
    outspec[1, :] = inspec[1, startpoint:endpoint + 1]

    return outspec
