from idl_lib import *
from idl_lib import __array__
import _global


def generateinfo(sn, srad, prad, atmoheight, fwhm, width, radialvelstart, radialvelend, semimajoraxis, inclination, nexposures, starfilename, exoplanetfilename, wlfilename):
    info = make_array(2, 15, string=True)

    info[:, 0] = __array__(('signal to noise        :', 'stellar radii (rsun)   :', 'planetary radii rjup   :', 'atmosphere height (km) :', 'fwhm                   :', 'gaussian width         :', 'radial velocity start  :',
                            'radial velocity end    :', 'semi-major axis        :', 'inclination            :', 'number of exposures    :', 'starfilename           :', 'exoplanetfilename      :', 'wlfilename             :'))
    info[:, 1] = __array__((string(sn), string(srad), string(prad), string(atmoheight), string(fwhm), string(width), string(radialvelstart), string(
        radialvelend), string(semimajoraxis), string(inclination), string(nexposures), starfilename, exoplanetfilename, wlfilename))

    return info
