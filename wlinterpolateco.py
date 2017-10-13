from idl_lib import *
from idl_lib import __array__
import _global

from bezier_interp import bezier_interp
from bezier_init import bezier_init


def wlinterpolateco(inspec, wlbefore, wlafter):

    if wlafter(0) == wlbefore(0) and wlafter(200) == wlbefore(200) and n_elements(wlafter) == n_elements(wlbefore):
        # 'identical WLscale'
        outspec = inspec

    else:
        if abs(wlbefore[2] / (wlbefore[2] - wlbefore[1])) > 5.1 * abs(wlafter[2] / (wlafter[2] - wlafter[1])):
            # 'convert to lower resolution'
            resfac = abs(wlbefore[2]) / abs(wlafter[2])
            spectemp = dblarr(ceil(n_elements(wlbefore)))
            wltemp = dblarr(ceil(n_elements(wlbefore)))
            for w in np.arange(1, n_elements(spectemp) + 1, 1):
                spectemp[w] = mean(inspec[floor(w * resfac)])
                wltemp[w] = wlbefore[floor(w + 0.5)]
            spectemp[0] = inspec[0]
            spectemp[n_elements(spectemp)] = inspec[n_elements(inspec)]

            outspec = bezier_interp(
                wltemp, spectemp, bezier_init(wltemp, spectemp), wlafter)

        else:
            # 'convert to approx equal resolution'
            outspec = bezier_interp(
                wlbefore, inspec, bezier_init(wlbefore, inspec), wlafter)

    return outspec
