import os.path
import numpy as np

from bezier_interp import bezier_interp

def readobservations(files, par, transitnumber, prefix, wlhr):
    filename = os.path.join(files.input,'cases','draft2','transit%d' % transitnumber, 'observations','obs%s.dat' % prefix)

    obstemp = np.loadtxt(filename)
    observations = bezier_interp(obstemp[:,0],obstemp[:,1:par.nexp+1],None,wlhr)

    return observations