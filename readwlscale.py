from idl_lib import *
from idl_lib import __array__
import _global


def readwlscale(pathname, wlfilename):
    
    filename = pathname  +  'wavelength/'  +  wlfilename  +  '.dat' 
    print(wlfilename) 
    
    wltemp = dblarr( 1 , 1e6 ) 
    s = double( 0 ) 
    filename = openr(2 , filename) 
    
    n = 0 
    ers = on_ioerror(ers) 
    while  n < 1e6 : 
        s = readf(2 , s) 
        wltemp[n] = s 
        n = n+1 
    
    if case(ers): 
        _ = close(all=True) 
    
    wl = wltemp[0:n-1+1] 
    
    return wl 
    return pathname, wlfilename 
