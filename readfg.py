from idl_lib import *
from idl_lib import __array__
import _global


def readfg(outpathname, inputfilename, nexp):

    filenamef = outpathname + 'data/f_' + inputfilename + '.dat'
    filenameg = outpathname + 'data/g_' + inputfilename + '.dat'
    filenamewl = outpathname + 'data/wl_' + inputfilename + '.dat'

    morethansize = 1e7
    wltemp = dblarr(1, morethansize)
    s = double(0)
    filenamewl = openr(5, filenamewl)
    n = 0

    while n < morethansize:
        s = readf(5)
        wltemp[n] = s
        n = n + 1

    wl = wltemp[0:n - 1 + 1]
    # WL=transpose(WL)

    ftemp = dblarr(nexp, n_elements(wl) + 1)
    s = dblarr(nexp, 1)
    filenamef = openr(5, filenamef)
    n = 0
    while n < morethansize:
        s = readf(5)
        ftemp[n, :] = s
        n = n + 1
    f = ftemp[0:n - 1 + 1, :]
    f = transpose(f)

    gtemp = dblarr(nexp, n_elements(wl) + 1)
    s = dblarr(nexp, 1)
    filenameg = openr(5, filenameg)
    n = 0
    while n < morethansize:
        s = readf(5)
        gtemp[n, :] = s
        n = n + 1
    print(n)
    g = gtemp[0:n - 1 + 1, :]
    g = transpose(g)

    return wl, f, g
