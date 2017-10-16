from idl_lib import dindgen

def radialvelocity(par):
    """ linear function from start velocity to end velocity """
    return (dindgen(par.nexposures)) / (par.nexposures - 1.) * \
        (par.radialvelend - par.radialvelstart) + par.radialvelstart
