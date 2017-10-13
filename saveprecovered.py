from idl_lib import *
from idl_lib import __array__
import _global


def saveprecovered(outpathname, inputfilename, solution, _lambda):

    filenameprec = outpathname + 'results/' + inputfilename + \
        '_' + strcompress(string(long(_lambda))) + '.dat'

    lun = 5
    fromatstring = '(' + string(long(2)) + '(d30.15,x))'
    fromatstring = strcompress(fromatstring, remove_all=True)
    openw(lun, filenameprec, get_lun=True, width=250)
    printf(lun, transpose(solution))
    close(lun)

    print('files saved os: outdata/' + inputfilename +
          '_' + strcompress(string(long(_lambda))) + '.dat')

    return 1
