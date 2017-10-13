from idl_lib import *
from idl_lib import __array__
import _global

from bezier_init import bezier_init
from bezier_interp import bezier_interp


def doppler(spectrumin, wl, velocity):
    c0 = 299792.
    wldoppler = wl * (sqrt((1. + velocity / c0)))

    wldoppler = __array__(
        (wldoppler[0] * 0.9, wldoppler, wldoppler[n_elements(wldoppler) - 1] * 1.1))
    si = __array__((1., spectrumin, 1.))
    # WLdoppler(0)=WLdoppler(0)*0.9
    # WLdoppler(n_elements(WLdoppler)-1)=WL(n_elements(WLdoppler)-1)*1.1
    # spectrumIn(0)=spectrumIn(0)*0.99
    # spectrumIn(n_elements(spectrumIn)-1)=spectrumIn(n_elements(spectrumIn)-1)*0.99

    spectrumout = bezier_interp(wldoppler, si, bezier_init(wldoppler, si), wl)
    # spectrumOut=WLinterpolateCO(spectrumIn,WLdoppler,WL)

    return spectrumout
