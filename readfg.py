import numpy as np


def readfg(files, n_exposures):
    """
    Read F, G, and Wavelength from Datafiles
    """

    filenamef = files.output + 'data/f_' + files.infile + '.dat'
    filenameg = files.output + 'data/g_' + files.infile + '.dat'
    filenamewl = files.output + 'data/wl_' + files.infile + '.dat'

    wl = np.loadtxt(filenamewl)
    #TODO unpack ?
    f = np.loadtxt(filenamef, ndmin=2)
    g = np.loadtxt(filenameg, ndmin=2)
    #TODO see above
    f = np.transpose(f)
    g = np.transpose(g)

    return wl, f, g
