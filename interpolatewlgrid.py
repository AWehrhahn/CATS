from idl_lib import *
from idl_lib import __array__
import _global

from bezier_init import bezier_init
from bezier_interp import bezier_interp

def interpolatewlgrid(inspec, wl, wlhr):

    # IS=cutSpec(Inspec,WL)
    i = inspec
    outspec = bezier_interp(wlhr, i, bezier_init(wlhr, i), wl)

    return outspec
