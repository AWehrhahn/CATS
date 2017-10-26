import os.path

from inputs import inputs
from readwlscale import readwlscale
from radialvelocity import radialvelocity
from planetvelocityco import planetvelocityco
from myvect import myvect
from readobservations import readobservations

def reconstructobservation(inputfile):
    files = lambda: None
    files.infile = inputfile
    files.path = os.path.expanduser('~/Documents/IDL/exoSpectro/NewSim/simulations/')
    files.input = files.path + 'indata/'
    files.output = files.path + 'outdata/'
    files.folder = inputfile

    # read master input: number of tranits, orbital parameters etc
    par, files = inputs(files)
    ntransits = 1
    
    wllr = readwlscale(files.input,files.wl)
    wlhr = readwlscale(files.input, files.wlhr)

    exptotal=0
    for transitnumber in range(1, ntransits+1):
        print('transit number: ', transitnumber)
        bvelocities = radialvelocity(par)
        pvelocities = planetvelocityco(par)

        par.my = myvect(par)

        print('Reading observation file')
        observation = readobservations(files, par, transitnumber, '', wlhr)