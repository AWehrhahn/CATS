from idl_lib import *
from idl_lib import __array__
import _global
from myatmocalc import myatmocalc
from intensityinterpolation import intensityinterpolation


def limbintatmoco(my_value, intspecall, srad, prad, atmoheight, wl):
    # number of points in the atmosphere
    np = 20.

    myatmo = myatmocalc(my_value, srad, prad, atmoheight, np)

    # create mean intensity spectra for all atmosphere points
    int = dblarr(n_elements(wl))
    for n in np.arange(0, np - 1 + 1, 1):
        int = int + intensityinterpolation(myatmo[n], intspecall)

    int = int / np

    return int
