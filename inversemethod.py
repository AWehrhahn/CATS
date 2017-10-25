from idl_lib import plot, oplot
import numpy as np
import os.path
import matplotlib.pyplot as plt

from cutawaybad import cutawaybad
from inputs import inputs
from readfg import readfg
from deltawavecreate import deltawavecreate
from eqvsys import eqvsys
from readplanet_sc import readplanet_sc
from saveprecovered import saveprecovered


def inversemethod(inputfilename, _lambda):
    files = lambda: None
    files.infile = inputfilename
    files.path = os.path.expanduser('~/Documents/IDL/exoSpectro/NewSim/simulations/')
    files.input = files.path + 'indata/'
    files.output = files.path + 'outdata/'

    # read master input: number of tranits, orbital parameters etc
    par, files = inputs(files)

    # load wavelength scale and equation system
    wl, f, g = readfg(files, par.nexposures)
    wl, f, g = cutawaybad(wl, f, g, par)

    # create deltaWL
    dwl2 = deltawavecreate(wl)

    # Calculate solution
    solution = eqvsys(f, g, wl, dwl2, _lambda)
    solution = solution / np.max(solution)

    # Read exo-atmosphere from datafile
    #exo = readplanet_sc(wl, files)

    # Plot
    #plt.plot(wl, exo)
    plt.plot(wl, solution, 'r')

    # Save Data
    solu = np.array([[wl], [solution]])
    saveprecovered(files, solu, _lambda)

    return solu
