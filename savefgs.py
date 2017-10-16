import numpy as np

def savefgs(files, f, g, wl, observation):

    filenameobs = files.output + 'data/obs_' + files.infile + '.dat'
    filenamef = files.output + 'data/f_' + files.infile + '.dat'
    filenameg = files.output + 'data/g_' + files.infile + '.dat'
    filenamewl = files.output + 'data/wl_' + files.infile + '.dat'

    np.savetxt(filenameobs, np.transpose(observation))
    np.savetxt(filenamef, np.transpose(f))
    np.savetxt(filenameg, np.transpose(g))
    np.savetxt(filenamewl, np.transpose(wl))