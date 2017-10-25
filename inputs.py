"""
Read data from configuration file
"""
import numpy as np
from idl_lib import common


def inputs(files):
    """ Read Data from inputfile in inpathname directory """

    filename = files.input + 'createObs/' + files.infile #+ '.txt'
    print(filename)
    inputsall = common(filename,2)[0]
    #Put all parameters into one object
    par = lambda: None

    # Signal to Noise
    par.sn = float(inputsall[1])
    # Stellar Radius
    par.srad = float(inputsall[3])
    # Planet Radius
    par.prad = float(inputsall[5])
    # Atmosphere height
    par.atmoheight = float(inputsall[7])
    # FWHM from Instrument
    par.fwhm = float(inputsall[9])
    # width, from instrument
    par.width = float(inputsall[11])
    # Radial Velocity at the start of transit
    par.radialvelstart = float(inputsall[13])
    # Radial Velocity at the end of transit
    par.radialvelend = float(inputsall[15])

    # Orbit Parameters
    par.semimajoraxis = float(inputsall[17])
    par.inclination = float(inputsall[19])
    par.period = float(inputsall[21])
    par.transitduration = float(inputsall[23])
    par.nexposures = float(inputsall[25])

    files.star = inputsall[27]
    files.exoplanet = inputsall[29]
    files.wl = inputsall[31]
    files.wlhr = inputsall[33]

    # convert all distaces to same units (km)
    rsun = 696000.
    rjup = 71350
    au = 149597871.
    secs = 24. * 60. * 60.
    
    par.srad = par.srad * rsun
    par.prad = par.prad * rjup
    par.semimajoraxis = par.semimajoraxis * au
    par.period = par.period * secs
    par.transitduration = par.transitduration * secs

    return par, files