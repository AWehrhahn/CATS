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
    pathname = '~/documents/idl/exospectro/newsim/simulations/'

    # read input parameters
    inpathname = pathname + 'indata/'
    outpathname = pathname + 'outdata/'
    sn, srad, prad, atmoheight, fwhm, width, radialvelstart, radialvelend, semimajoraxis, inclination, nexposures, starfilename, exoplanetfilename, wlfilename, wlhrfilename, period, transitduration = inputs(
        inputfilename, inpathname)

    # create my-vector
    my = myvect(srad, prad, atmoheight, semimajoraxis, inclination, nexposures)

    # create velocity vector
    bvelocities = radialvelocity(radialvelstart, radialvelend, nexposures)

    # create planet radial velocity from circular motion
    pvelocities = planetvelocityco(
        inclination, nexposures, semimajoraxis, period, srad, prad, transitduration)

    # load wave lenght scale
    wllr = readwlscale(inpathname, wlfilename)
    wlhr = readwlscale(inpathname, wlhrfilename)

    # load exoplanet spectrums
    print('reading exoplanet spectrum')
    planetspec = readplanet_sc(inpathname, exoplanetfilename, wlhr)

    # load telluric spectrum
    # print,'Reading Telluric spectrum'
    # TelluricSpec=readTelluric_SC(InPathName,TelluricFilename,WLhr)

    # load stellar spectrum
    print('reading stellar spectra')
    normal, fluxspec, intspecall = readstar_marcs(inpathname, starfilename,
                                                  wlhr)

    # create each individual exposure, one at a time
    observation = dblarr(n_elements(wllr), nexposures)
    normal = dblarr(nexposures)
    for n in np.arange(0, nexposures, 1):
        print('create exposure number: ' + string(n + 1) +
              ' of ' + string(long(nexposures)))

        observation2, norma = exposurecreate(fluxspec, intspecall, planetspec, wlhr, bvelocities,
                                             pvelocities, my, n, srad, prad, atmoheight, wllr, sn, fwhm, width)
        observation[n, :] = observation2
        normal[n] = norma

    # DATA ANALYSIS
    print('data analysis')

    # create F and G for each exposure
    f = dblarr(n_elements(wllr), nexposures)
    g = dblarr(n_elements(wllr), nexposures)
    st = dblarr(n_elements(wllr), nexposures)
    for n in np.arange(0, nexposures - 1 + 1, 1):
        print('recreate exposure number: ' + string(n + 1) +
              ' of ' + string(long(nexposures)))
        obsspec = observation[n, :]

        ggg, fff, stack = exposurerecreate(fluxspec, intspecall, obsspec, prad, srad, atmoheight, wlhr, wllr, bvelocities,
                                           pvelocities, my, n, fwhm, width)
        f[n, :] = fff
        g[n, :] = ggg
        st[n, :] = stack

    # save F and G and WLlr
    savefgs(outpathname, inputfilename,
            nexposures, f, g, wllr, observation)

    _lambda = 1000.
    solution = inversemethod(inputfilename, _lambda)
    plot(wlhr, fluxspec)
    oplot(wlhr, planetspec, color=155)
    oplot(solution[0, :], solution[1, :], color=255)
    solution = help(solution)

    return 1
