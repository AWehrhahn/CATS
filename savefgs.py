from idl_lib import *
from idl_lib import __array__
import _global


def savefgs(outpathname, inputfilename, exptotal, f, g, wl, obs):
    
    filenameobs = outpathname  +  'data/obs_'  +  inputfilename  +  '.dat' 
    filenamef = outpathname  +  'data/f_'  +  inputfilename  +  '.dat' 
    filenameg = outpathname  +  'data/g_'  +  inputfilename  +  '.dat' 
    filenamewl = outpathname  +  'data/wl_'  +  inputfilename  +  '.dat' 
    
    lun = 3 
    fromatstring = '(' +string(long( exptotal )) +  '(d30.15,x))' 
    # fromatstring='('+string(long(ExpTotal))+'(D20.15,x))'
    fromatstring = strcompress(fromatstring, remove_all=True) 
    openw(lun, filenamef, get_lun=True, width=250) 
    printf(lun ,transpose( obs ), format=fromatstring) 
    close(all=True) 
    
    
    lun = 3 
    fromatstring = '(' +string(long( exptotal )) +  '(d30.15,x))' 
    # fromatstring='('+string(long(ExpTotal))+'(D20.15,x))'
    fromatstring = strcompress(fromatstring, remove_all=True) 
    openw(lun, filenamef, get_lun=True, width=250) 
    printf(lun ,transpose( f ), format=fromatstring) 
    close(all=True) 
    
    lun = 4 
    fromatstring = '(' +string(long( exptotal )) +  '(d30.15,x))' 
    fromatstring = strcompress(fromatstring, remove_all=True) 
    openw(lun, filenameg, get_lun=True, width=250) 
    printf(lun ,transpose( g ), format=fromatstring) 
    close(all=True) 
    
    lun = 5 
    fromatstring = '(' +string(long( 1 )) +  '(d30.15,x))' 
    fromatstring = strcompress(fromatstring, remove_all=True) 
    openw(lun, filenamewl, get_lun=True, width=250) 
    printf(lun ,transpose( wl ), format=fromatstring) 
    close(all=True) 
    
    
    return 1 
    