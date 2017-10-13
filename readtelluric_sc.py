from idl_lib import *
from idl_lib import __array__
import _global


def readtelluric_sc(pathname, telluricfilename, wl):
    
    filename = pathname  +  'telluric/'  +  telluricfilename  +  '.dat' 
    
    
    
    filelength = 194187. 
    indata = dblarr( 2 , filelength ) 
    filename = openr(1 , filename) 
    indata = readf(1 , indata) 
    _ = close(all=True) 
    
    # PlanetSpec=WLinterpolateCO(indata(1,*),indata(0,*),WL)
    tellspect = abs(transpose(indata[:, 1])) 
    wlt = abs(transpose(indata[:, 0])) 
    tellspec = wlinterpolateco( tellspect , wlt , wl ) 
    
    return tellspec 
    return pathname, telluricfilename, wl 

