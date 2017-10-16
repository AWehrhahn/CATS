"""
Read data from configuration file
"""
import numpy as np


def inputs(files):
    """ Read Data from inputfile in inpathname directory """

    filename = files.input + 'createobs/' + files.infile + '.txt'
    print(filename)
    inputsall = np.loadtxt(filename)
    #Put all parameters into one object
    par = lambda: None

    # Signal to Noise
    par.sn = inputsall[1]
    # Stellar Radius
    par.srad = inputsall[3]
    # Planet Radius
    par.prad = (inputsall[5])
    # Atmosphere height
    par.atmoheight = (inputsall[7])
    # FWHM from Instrument
    par.fwhm = (inputsall[9])
    # width, from instrument
    par.width = (inputsall[11])
    # Radial Velocity at the start of transit
    par.radialvelstart = (inputsall[13])
    # Radial Velocity at the end of transit
    par.radialvelend = (inputsall[15])

    # Orbit Parameters
    par.semimajoraxis = (inputsall[17])
    par.inclination = (inputsall[19])
    par.period = (inputsall[21])
    par.transitduration = (inputsall[23])
    par.nexposures = (inputsall[25])

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