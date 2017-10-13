from idl_lib import *
from idl_lib import __array__
import _global


def instprof(inspec, fwhm, width):

    height = 0.08
    x = dindgen(width * 2. + 1.)
    x = x - width

    y = height * exp(-(x)**2. / 2. * (2.67 / fwhm) ** 2.)

    # Int = INT_TABULATED(x, y)

    # make extentions for the ends of the new spectrum and set ends to equal to endpoint to not change results much.
    extentionbeg = dblarr(width) + inspec[0]
    extentionend = dblarr(width) + inspec[n_elements(inspec)]
    # new spectrum combined with the extentions at the ends.
    extspec = __array__((extentionbeg, inspec, extentionend))

    # makes one spectrum for each convolution loop and adds them together to the

    outspec = dblarr(n_elements(inspec))  # makes an empty array to add to.
    # g=n*A
    for i in range(n_elements(inspec) + 1):
        outspec[i] = total(extspec[i:i + 2. * width + 1] * y)

    # normalize, detected number of photons unchanged
    outspec = outspec / total(outspec) * total(inspec)

    return outspec
