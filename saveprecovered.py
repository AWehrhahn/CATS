import numpy as np
import os.path

from idl_lib import strcompress


def saveprecovered(files, solution, _lambda):

    filenameprec = os.path.join(
        files.output, 'results', files.infile, '_' + str(_lambda) + '.dat')

    if not os.path.exists(os.path.dirname(filenameprec)):
        os.makedirs(os.path.dirname(filenameprec))
    with open(filenameprec, 'wb') as f:
        np.savetxt(f, np.column_stack(solution))

    print('files saved os: ', filenameprec)
