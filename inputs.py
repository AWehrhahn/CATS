from idl_lib import *
from idl_lib import __array__
import _global


def inputs(inputfilename, inpathname):
    """ Read Data from inputfile in inpathname directory """

    filename = inpathname + 'createobs/' + inputfilename + '.txt'
    print(filename)
    filelength = 34
    inputsall = make_array(1, filelength, string=True)
    openr(1, filename)
    inputsall = readf(1)
    close(1)

    sn = double(inputsall[1])
    srad = double(inputsall[3])
    prad = double(inputsall[5])
    atmoheight = double(inputsall[7])
    fwhm = double(inputsall[9])
    width = double(inputsall[11])
    radialvelstart = double(inputsall[13])
    radialvelend = double(inputsall[15])
    semimajoraxis = double(inputsall[17])
    inclination = double(inputsall[19])
    period = double(inputsall[21])
    transitduration = double(inputsall[23])
    nexposures = double(inputsall[25])
    starfilename = inputsall[27]
    exoplanetfilename = inputsall[29]
    wlfilename = inputsall[31]
    wlhrfilename = inputsall[33]

    # convert all distaces to same units (km)
    rsun = 696000.
    rjup = 71350
    au = 149597871.
    secs = 24. * 60. * 60.
    srad = srad * rsun
    prad = prad * rjup
    semimajoraxis = semimajoraxis * au
    period = period * secs
    transitduration = transitduration * secs

    return sn, srad, prad, atmoheight, fwhm, width, radialvelstart, radialvelend, semimajoraxis, inclination, nexposures, starfilename, exoplanetfilename, wlfilename, wlhrfilename, period, transitduration
