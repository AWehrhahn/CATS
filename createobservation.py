from idl_lib import *
from idl_lib import __array__
import _global

from inputs import inputs
from myvect import myvect
from radialvelocity import radialvelocity
from planetvelocityco import planetvelocityco
from readwlscale import readwlscale
from readplanet_sc import readplanet_sc
from readstar_marcs import readstar_marcs
from exposurecreate import exposurecreate
from exposurerecreate import exposurerecreate
from savefgs import savefgs
from inversemethod import inversemethod


def createobservation(inputfilename):
    """ CREATE OBSERVATIONS """
    files = lambda: None
    files.infile = inputfilename
    files.path = '~/documents/idl/exospectro/newsim/simulations/'

    # read input parameters
    files.input = files.pathname + 'indata/'
    files.outpathname = files.pathname + 'outdata/'
    par, files = inputs(files)

    # create my-vector
    par.my = myvect(par)

    # create velocity vector
    par.bvelocities = radialvelocity(par)

    # create planet radial velocity from circular motion
    par.pvelocities = planetvelocityco(par)

    # load wave lenght scale
    wllr = readwlscale(files.path, files.wl)
    wlhr = readwlscale(files.path, files.wlhr)

    # load exoplanet spectrums
    print('reading exoplanet spectrum')
    planetspec = readplanet_sc(files, wlhr)

    # load telluric spectrum
    # print,'Reading Telluric spectrum'
    # TelluricSpec=readTelluric_SC(InPathName,TelluricFilename,WLhr)

    # load stellar spectrum
    print('reading stellar spectra')
    normal, fluxspec, intspecall = readstar_marcs(files, wlhr)

    # create each individual exposure, one at a time
    observation = dblarr(n_elements(wllr), par.nexposures)
    normal = dblarr(par.nexposures) 
    for n in range(par.nexposures):
        print('create exposure number: ' + str(n + 1) +
              ' of ' + str(par.nexposures))

        observation2, norma = exposurecreate(fluxspec, intspecall, planetspec, wlhr,wllr, n, par)
        observation[n, :] = observation2
        normal[n] = norma

    # DATA ANALYSIS
    print('data analysis')

    # create F and G for each exposure
    f = dblarr(n_elements(wllr), par.nexposures)
    g = dblarr(n_elements(wllr), par.nexposures)
    st = dblarr(n_elements(wllr), par.nexposures)
    for n in range(par.nexposures):
        print('recreate exposure number: ' + string(n + 1) +
              ' of ' + string(par.nexposures))
        obsspec = observation[n, :]

        ggg, fff, stack = exposurerecreate(fluxspec, intspecall, obsspec, wlhr, wllr, n, par)
        f[n, :] = fff
        g[n, :] = ggg
        st[n, :] = stack

    # save F and G and WLlr
    savefgs(files, f, g, wllr, observation)

    _lambda = 1000.
    solution = inversemethod(inputfilename, _lambda)
    plot(wlhr, fluxspec)
    oplot(wlhr, planetspec, color=155)
    oplot(solution[0, :], solution[1, :], color=255)
    solution = help(solution)

    return 1
