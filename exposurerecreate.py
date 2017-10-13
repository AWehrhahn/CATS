from idl_lib import *
from idl_lib import __array__
import _global

from limbintplanetco import limbintplanetco
from doppler import doppler
from wlinterpolateco import wlinterpolateco
from instprof import instprof
from addnoise import addnoise


def exposurerecreate(fluxspec, intspecall, obsspec, prad, srad, atmoheight, wlhr, wllr, bvelocities, pvelocities, my, n, fwhm, width):
    # calculate areas
    sarea = _global.pi * srad**2.
    parea = _global.pi * prad**2.
    atmoarea = _global.pi * (prad + atmoheight) ** 2. - parea
    parea = parea / sarea
    atmoarea = atmoarea / sarea
    paarea = parea + atmoarea

    # current MYvalue
    my_value = my[n]
    bvelocity = bvelocities[n]
    pvelocity = pvelocities[n]

    # calculate what intensity spectrum to use for this exposure
    intspecplanet = limbintplanetco(
        my_value, intspecall, srad, prad, atmoheight, wlhr)

    fluxspecshifted = doppler(fluxspec, wlhr, -pvelocity)
    intspecplanetshifted = doppler(intspecplanet, wlhr, -pvelocity)
    # tellSpecShifted=doppler(TelluricSpec,WLhr,-(Bvelocity+Pvelocity))
    obsspechr = wlinterpolateco(obsspec, wllr, wlhr)
    obsspecshifted = doppler(obsspechr, wlhr, -(bvelocity + pvelocity))

    # interpolate on correct WL-grid
    flt = wlinterpolateco(fluxspecshifted, wlhr, wllr)
    ipt = wlinterpolateco(intspecplanetshifted, wlhr, wllr)
    # Tet=WLinterpolateCO(tellSpecShifted,WLhr,WLlr)
    ob = wlinterpolateco(obsspecshifted, wlhr, wllr)

    # instrumental profile
    fl = instprof(flt, fwhm, width)
    ip = instprof(ipt, fwhm, width)
    # Te=instProf(Tet,FWHM,width)

    # anti-normalization
    # normalization2=1.-median(Ip/Fl)*PAarea
    # normalization3=1.-max(Ip/Fl)*Parea
    # normalization=(normalization2+normalization3)*0.5
    fl = abs(fl)
    ip = abs(ip)
    # Te=abs(Te)
    ob = abs(ob)  # *normalization

    # for equation system
    # GGG = Te
    fff = ((ob - fl + paarea * ip) / ip) * (1. / atmoarea)
    ggg = fff / fff

    stack = (ob - fl + ip * paarea) / (ip)
    plot(wllr, ob, xrange=[
         wllr[(floor(n_elements(wllr) - 1)) * 0.1], wllr[0]], yrange=[1.2, 0])
    oplot(wllr, fl, color=75)
    # oplot,wllr,te,color=125
    oplot(wllr, stack + 1., color=255)

    return ggg, fff, stack
