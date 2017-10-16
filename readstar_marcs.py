from idl_lib import dblarr
import numpy as np
from wlinterpolateco import wlinterpolateco


def readstar_marcs(files, wl):

    filename = [i for i in range(11)]
    filename[0] = files.path + 'stars/' + files.star + '.flx'
    filename[1] = files.path + 'stars/' + files.star + '0.1' + '.int'
    filename[2] = files.path + 'stars/' + files.star + '0.2' + '.int'
    filename[3] = files.path + 'stars/' + files.star + '0.3' + '.int'
    filename[4] = files.path + 'stars/' + files.star + '0.4' + '.int'
    filename[5] = files.path + 'stars/' + files.star + '0.5' + '.int'
    filename[6] = files.path + 'stars/' + files.star + '0.6' + '.int'
    filename[7] = files.path + 'stars/' + files.star + '0.7' + '.int'
    filename[8] = files.path + 'stars/' + files.star + '0.8' + '.int'
    filename[9] = files.path + 'stars/' + files.star + '0.9' + '.int'
    filename[10] = files.path + 'stars/' + files.star + '1.0' + '.int'

    # flux
    indata = np.loadtxt(filename[0], ndmin=2)

    fluxspect = abs(np.transpose(indata[:, 2]))
    normalt = abs(np.transpose(indata[:, 2]))
    fluxspect = fluxspect / normalt
    wlt = abs(np.transpose(indata[:, 0]))
    fluxspec = wlinterpolateco(fluxspect, wlt, wl)
    normal = wlinterpolateco(normalt, wlt, wl)

    # intensity
    intspecall = dblarr(len(wl), len(filename))

    for i in range(1, len(filename)):
        indata = np.loadtxt(filename[i], ndmin=2)
        intspect = abs(np.transpose(indata[:, 1])) / normalt
        wlt = abs(np.transpose(indata[:, 0]))
        intspec = wlinterpolateco(intspect, wlt, wl)

        intspecall[i, :] = intspec

    return normal, fluxspec, intspecall
