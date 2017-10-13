from idl_lib import *
from idl_lib import __array__
import _global

from cutawaybad import cutawaybad
from inputs import inputs
from readfg import readfg
from deltawavecreate import deltawavecreate
from eqvsys import eqvsys
from readplanet_sc import readplanet_sc
from saveprecovered import saveprecovered


def inversemethod(inputfilename, _lambda):
    pathname = '~/documents/idl/exospectro/newsim/simulations/'
    inpathname = pathname + 'indata/'
    outpathname = pathname + 'outdata/'

    # read master input: number of tranits, orbital parameters etc
    sn, srad, prad, atmoheight, fwhm, width, radialvelstart, radialvelend,    semimajoraxis, inclination, nexposures, starfilename, exoplanetfilename, wlfilename, wlhrfilename, period, transitduration = inputs(
        inputfilename, inpathname)

    # load wave lenght scale and equation system
    wl, f, g = readfg(outpathname, inputfilename, nexposures)
    # help,wl
    wl, f, g = cutawaybad(nexposures, wl, f, g, radialvelstart, radialvelend,
                          semimajoraxis, period, srad, prad, inclination, transitduration)
    # help,wl

    # create deltaWL
    dwl2 = deltawavecreate(wl)

    solution = eqvsys(f, g, wl, dwl2, _lambda)

    solution = solution - min(solution) + 0.5  # normalize
    solution = solution / max(solution)

    exo = readplanet_sc(inpathname, exoplanetfilename, wl)

    plot(wl, exo)  # ,yrange = [0., 1.1]
    oplot(wl, solution, color=255)

    solu = __array__(((wl), (solution)))

    saveprecovered(outpathname, inputfilename, solu, _lambda)

    return solu
