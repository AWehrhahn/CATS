from idl_lib import *
from idl_lib import __array__
import _global

from numpy import pi

from limbintplanetco import limbintplanetco
from doppler import doppler
from wlinterpolateco import wlinterpolateco
from instprof import instprof
from addnoise import addnoise


def exposurecreate(fluxspec, intspecall, planetspec, wlhr,wllr, n, par):
    # calculate areas
    sarea = pi * par.srad**2.
    parea = pi * par.prad**2.
    atmoarea = pi * (par.prad + par.atmoheight) ** 2. - parea
    parea = parea / sarea
    atmoarea = atmoarea / sarea
    paarea = parea + atmoarea

    # current MYvalue
    my_value = par.my[n]
    bvelocity = par.bvelocities[n]
    pvelocity = par.pvelocities[n]

    # calculate what intensity spectrum to use for this exposure
    intspecplanet = limbintplanetco(my_value, intspecall, wlhr, par)

    # doppler shift spectra (not telluric)
    fluxspecshifted = doppler(fluxspec, wlhr, bvelocity)
    intspecplanetshifted = doppler(intspecplanet, wlhr, bvelocity)
    planetspecshifted = doppler(planetspec, wlhr, bvelocity + pvelocity)

    fluxspecshifted = wlinterpolateco(fluxspecshifted, wlhr, wllr)
    intspecplanetshifted = wlinterpolateco(intspecplanetshifted, wlhr, wllr)
    planetspecshifted = wlinterpolateco(planetspecshifted, wlhr, wllr)
    # TelluSpec=WLinterpolateCO(TelluricSpec,wlhr,wllr)

    # instrumental profile
    fl = abs(instprof(fluxspecshifted, fwhm, width))
    ip = abs(instprof(intspecplanetshifted, fwhm, width))
    # Te=abs(instProf(TelluSpec,FWHM,width))
    pl = abs(instprof(planetspecshifted, fwhm, width))

    # multiply
    obslr = ((fl) - (paarea * ip) + (atmoarea * ip * pl))

    # normalize
    norma = max(obslr)
    # ObsLR=ObsLR/max(ObsLR)

    # add noise
    obs = addnoise(obslr, sn)

    plot(wllr, obs, xrange=[
         wllr[(floor(n_elements(wllr) - 1)) * 0.1], wllr[0]], yrange=[1.2, 0])
    oplot(wllr, fluxspecshifted, color=75)
    # oplot,wllr,TelluSpec,color=125
    oplot(wllr, planetspecshifted, color=255)

    return obs, norma
