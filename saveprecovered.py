import numpy as np
import os.path

from idl_lib import strcompress


def saveprecovered(files, solution, _lambda):

    filenameprec = os.path.join(
        files.output, 'results', files.infile, '_' + _lambda + '.dat')

    np.savetxt(filenameprec, np.transpose(solution))

    print('files saved os: ', filenameprec)
