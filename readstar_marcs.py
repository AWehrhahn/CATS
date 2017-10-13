from idl_lib import *
from idl_lib import __array__
import _global

from wlinterpolateco import wlinterpolateco


def readstar_marcs(pathname, starfilename, wl):

    filename = make_array(1, 11, string=True)
    filename[0] = pathname + 'stars/' + starfilename + '.flx'
    filename[1] = pathname + 'stars/' + starfilename + '0.1' + '.int'
    filename[2] = pathname + 'stars/' + starfilename + '0.2' + '.int'
    filename[3] = pathname + 'stars/' + starfilename + '0.3' + '.int'
    filename[4] = pathname + 'stars/' + starfilename + '0.4' + '.int'
    filename[5] = pathname + 'stars/' + starfilename + '0.5' + '.int'
    filename[6] = pathname + 'stars/' + starfilename + '0.6' + '.int'
    filename[7] = pathname + 'stars/' + starfilename + '0.7' + '.int'
    filename[8] = pathname + 'stars/' + starfilename + '0.8' + '.int'
    filename[9] = pathname + 'stars/' + starfilename + '0.9' + '.int'
    filename[10] = pathname + 'stars/' + starfilename + '1.0' + '.int'

    # flux
    filelength = 194187.
    indata = dblarr(5, filelength)
    filename[0] = openr(1, filename[0])
    indata = readf(1)
    close(1)

    fluxspect = abs(transpose(indata[:, 2]))
    normalt = abs(transpose(indata[:, 2]))
    fluxspect = fluxspect / normalt
    wlt = abs(transpose(indata[:, 0]))
    fluxspec = wlinterpolateco(fluxspect, wlt, wl)
    normal = wlinterpolateco(normalt, wlt, wl)

    # intensity
    intspecall = dblarr(n_elements(wl), 11)

    for i in np.arange(1, 10 + 1, 1):
        indata = dblarr(2, filelength)
        filename[i] = openr(1, filename[i])
        indata = readf(1)
        close(1)

        intspect = abs(transpose(indata[:, 1])) / normalt
        wlt = abs(transpose(indata[:, 0]))
        intspec = wlinterpolateco(intspect, wlt, wl)

        intspecall[i, :] = intspec

    return normal, fluxspec, intspecall
