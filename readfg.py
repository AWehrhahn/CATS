import numpy as np


def readfg(files, n_exposures):
    """
    Read F, G, and Wavelength from Datafiles
    Phi = sum(G*P - F)
    F = - Observation + Flux_model * Telluric + (Radius_planet/Radius_star)**2 * Intensity_planet_model * Telluric
    G = Intensity_atmosphere_model * Telluric
    """

    filenamef = files.output + 'data/f_' + files.infile + '.dat'
    filenameg = files.output + 'data/g_' + files.infile + '.dat'
    filenamewl = files.output + 'data/wl_' + files.infile + '.dat'

    wl = np.loadtxt(filenamewl, delimiter=',')
    #TODO unpack ?
    f = np.loadtxt(filenamef, ndmin=2, delimiter=',')
    g = np.loadtxt(filenameg, ndmin=2, delimiter=',')
    #TODO see above
    f = np.transpose(f)
    g = np.transpose(g)

    return wl, f, g
