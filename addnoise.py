from idl_lib import *
from idl_lib import __array__
import _global


def addnoise(inspec, sn):

    # RandSeed = 252151.+ 	;random number seed, set to "seed" if random, set to constant if you want to use the same noise each time

    # noise=1.+(1./SN)*randomn(RandSeed,n_elements(inSpec))
    noise = 1. + (1. / sn) * randomn(n_elements(inspec))
    outspec = inspec * noise
    # print,'non random noise'
    return outspec
