"""
A collection of idl functions, translated to python
"""
import os
import fnmatch
import datetime
import regex as re
import pickle
import glob
import types
import subprocess
#import pydl
import numpy as np
import scipy.io
import scipy.optimize
import scipy.interpolate
import scipy.ndimage
import matplotlib.pyplot as plt
import _global

"""
--------------
    Basics
--------------
"""


def goto(*args):
    """ OBSOLETE, KILL IT WITH FIRE """
    print('GOTO found, please change idl code')
    raise Exception


def stop(msg):
    print(msg)
    raise AttributeError


def message(msg, **kwargs):
    """ print a message to the screen """
    print(msg)
    return ''


def arg_present(var):
    return True


def call_function(name, *args):
    module = __import__('REDUCE.' + name)
    method = getattr(module, name)
    method = getattr(method, name)
    return method(*args)


def call_procedure(name, *args):
    return call_function(name, *args)


def spawn(*args, **kwargs):
    return subprocess.call(*args)


def keyword_set(keyword):
    """
    All IDL keywords are converted to None by IDL_to_Python
    keyword_set then simply becomes a comparison of the keyword to None
    """
    return keyword is not None


def tag_exist(var, tag):
    return var.hasattr(tag)


def tag_names(var):
    return vars(var).keys()


def n_elements(arr):
    """ the IDL version of len, but in multiple dimensions """
    if arr is None:
        return 0
    try:
        return np.size(arr)
    except ValueError:
        return len(arr)


def n_params():
    """ 
    number of parameters passed to the function 
    Obsolete in Python, therefore assume, that enough parameters are passed
    """
    return 100000


def on_error(*args):
    """ Obsolete """
    pass


def systime(**kwargs):
    return datetime.datetime.now()


def restore(file):
    try:
        return scipy.io.readsav(file)
    except:
        with open(file, 'r') as f:
            return pickle.load(f)


def save(file, *args):
    with open(file, 'w') as f:
        pickle.dump(f, args)


def struct(*args):
    # return an empty strcuture, values have to be fixed manually
    return lambda: None
    return args


def create_struct(*args):
    # empty struct
    def s(): return None
    i = -1
    while i < len(args):
        i += 1
        if isinstance(args[i], str):
            setattr(s, args[i], args[i + 1])
            i += 1
            continue
        elif isinstance(args[i], list):
            for elem, j in zip(args[i], range(len(args[i]))):
                setattr(s, elem, args[i + j])
            i += len(args[i])
            continue
        elif isinstance(args[i], object):
            attr = vars(args[i])
            for key, value in attr.items():
                setattr(s, key, value)
            continue
    return s


def defsysv(name, value):
    setattr(_global, name, value)


"""
----------
"""


def file_mkdir(dir):
    os.makedirs(dir)
    return dir


def file_test(file, directory=False):
    if directory:
        os.path.isdir(file)
    return os.path.exists(file)


def file_search(specification, rec_pattern=None, test_directory=False, mark_directory=False, count=None, **kwargs):
    if rec_pattern is None:
        res = glob.glob(specification)
    else:
        specification = os.path.join(specification, rec_pattern)
        res = glob.glob(specification, recursive=True)

    if test_directory:
        res = [r for r in res if os.path.isdir(r)]

    if mark_directory:
        res = [r + os.sep if os.path.isdir(r) else r for r in res]

    res = np.array(res)
    if count is not None:
        return res, len(res)
    return res


def file_basename(file):
    res = os.path.basename(file)
    if res == '':
        return os.path.basename(os.path.dirname(file))
    else:
        return res


def findfile(specification, count=None):
    return file_search(specification, count=count)


"""
--------------
    Random
--------------
"""


def randomu(x):
    pass
    # return np.random(x)

def randomn(x):
    np.random.randn(x)
    pass

"""
------------
    Data
------------
"""


def cd(current=None):
    """ get current working directory """
    return os.getcwd()


lun = {}


def openw(unit, filename, **kwargs):
    """ open file to write """
    """
    i = -1
    while True:
        i += 1
        if i not in lun.keys():
            unit = i
            break
    """
    lun[unit] = filename
    return unit, lun[unit]


def openr(unit, filename, error=None, get_lun=False, **kwargs):
    """ open file to read """
    lun[unit] = filename
    return unit, lun[unit], error


def writeu(unit, *args):
    """ write raw binary data to file """
    with open(lun[unit], 'wb') as file:
        for a in args:
            file.write(bytes(a))
    return unit


def readu(unit, *args):
    """ read binary data """
    with open(lun[unit], 'rb') as file:
        return file.read()


def printf(unit, *args):
    with open(lun[unit], 'w') as file:
        file.write(*args)


def readf(unit):
    with open(lun[unit], 'r') as file:
        return file.read()


def close(unit):
    """ 'close' file with unit unit """
    #del lun[unit]
    return unit


def free_lun(unit):
    del lun[unit]
    return unit


def point_lun(unit, new):
    lun[unit] = lun[new]


"""
------------
    Math
------------
"""


def abs(*args):
    """ idl version of math.abs """
    return np.abs(*args)


def ceil(*args):
    """ ceiling of arg """
    return int(np.ceil(*args))


def cir_3pnt(x, y, r=None, x0=None, y0=None):
    """ radius and center of circle with points given by x and y """
    p1 = x[0] + y[0] * 1j
    p2 = x[1] + y[0] * 1j
    p3 = x[2] + y[2] * 1j
    w = p3 - p1
    w /= p2 - p1
    c = (p1 - p2) * (w - abs(w)**2) / 2j / w.imag - p1
    r = abs(c + p1)
    x0 = c.real
    y0 = c.imag
    return r, x0, y0


def complexround(x):
    """ round complex value """
    return np.round(x)


def round(x):
    """ round value """
    return int(np.round(x))


def diag_matrix(a, diag=0):
    """ construc diagonal matrix from input vector a, or vice versa """
    return np.diag(a, k=diag)


def dist(n, m=None):
    """ sth about arrays and frequency """
    raise NotImplementedError


def exp(x):
    """ exponetial e**x """
    return np.exp(x)


def floor(x):
    """ floor of x """
    return int(np.floor(x))


def matrix_multiply(a, b):
    """ dot product of a and b """
    return np.dot(b, a)


def sqrt(x):
    return np.sqrt(x)


def product(x, **kwargs):
    return np.prod(x)


def median(x, dimension=None):
    if dimension is not None:
        dimension -= 1
    else:
        dimension = 0

    return np.median(x, axis=dimension)


"""
--------------
    Trig
--------------
"""


def sin(x):
    return np.sin(x)


def cos(x):
    return np.cos(x)


def tan(x):
    return np.tan(x)


def asin(x):
    return np.arcsin(x)


def acos(x):
    return np.arccos(x)


def atan(x):
    return np.arctan(x)


def sinh(x):
    return np.sinh(x)


def cosh(x):
    return np.cosh(x)


def tanh(x):
    return np.tanh(x)


def asinh(x):
    return np.arcsinh(x)


def acosh(x):
    return np.arccosh(x)


def atanh(x):
    return np.arctanh(x)


"""
--------
"""

def value_locate(arr, val):
    return arr.index(val)


def array_equal(arr, val):
    """ True if all elemnts of arr are equal to val """
    return all(arr == val)


def array_indices(arr, ind):
    """ convert 1D indices into MD indices """
    raise NotImplementedError


def invert(arr):
    """ inverse of arr """
    return np.invert(arr)


def max(arg, count=None):
    """ idl max returns number of indices as well """
    if count is not None:
        if '=' not in count:
            count = len(arg)
        else:
            k, count = count.split('=')
            if k.strip() == 'min':
                count = min(arg)
    if count is not None:
        return np.max(arg), count
    return np.max(arg)


def min(arg, count=None):
    """ idl min returns number of indices as well """
    if count is not None:
        if '=' not in count:
            count = len(arg)
        else:
            k, count = count.split('=')
            if k.strip() == 'max':
                count = max(arg)
    if count is not None:
        return np.min(arg), count
    return np.min(arg)


def minmax(arg):
    """ return min and max or arg """
    return [min(arg), max(arg)]


def reform(arr, *args):
    """ reshape dimensions of arr """
    return arr.reshape(args)


def replicate_inplace(arr, val):
    return replicate(val, *np.shape(arr))


def reverse(arr):
    """ reverse arr """
    return arr[::-1]


def signum(x):
    """ sign of x """
    return np.sign(x)


def size(x, tname=False, n_dimension=False):
    """ idl.size = np.shape """
    # dim dim1 dim2 ... typecode elemtotal
    if isinstance(x, np.ndarray):
        dim = x.ndim
        shape = np.shape(x)[::-1]
        t = 5  # maybe check?
        elem = x.size
    else:  # probably a header
        dim = 1
        shape = (len(x),)
        t = 7  # for string
        elem = len(x)
    res = [dim, *shape, t, elem]

    if n_dimension:
        res = dim
    if tname:
        res = str(res)
    return res


def sort(arr):
    """ sort an array """
    return np.argsort(arr)


def total(*args):
    """ idl version of sum """
    return np.sum(*args)


def transpose(arr):
    """ transpose of arr """
    return np.transpose(arr)


def uniq(arr):
    """ return a uniq array """
    return np.unique(arr, return_index=True)[1]


def where(cond, count=None, null=False):
    """
    Returns a mask for an array with indices that fulfill the given condition
    Count is the name of a variable that will hold the number of elements
    Its not good python code, but it is working python code
    """
    x = np.where(cond)[0]
    c = len(x)
    if len(x) == 0:
        if not null:
            x = -1
        else:
            x = [False for i in cond]
    if count is not None:
        return x, c
    return x


"""
-----
"""


def alog10(x):
    """ log to base 10 of x """
    return np.log10(x)


def interpolate(p, x, y=None):
    """ interpolate from p to postions given by x, y, z """
    #scipy.interpolate.griddata(p, )
    return np.interp(x, range(len(p)), p)


def rotate(arr, direction):
    return np.rot90(arr, -direction)


def temporary(x):
    return x


def trisol(a,b,c,d):
    '''
    from https://gist.github.com/cbellei/8ab3ab8551b8dfc8b081c518ccd9ada9
    TDMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    '''
    nf = len(d)  # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d))  # copy arrays
    for it in range(1, nf):
        mc = ac[it - 1] / bc[it - 1]
        bc[it] = bc[it] - mc * cc[it - 1]
        dc[it] = dc[it] - mc * dc[it - 1]

    xc = bc
    xc[-1] = dc[-1] / bc[-1]

    for il in range(nf - 2, -1, -1):
        xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]

    return xc


"""
-------------------
    Data types
-------------------
"""

type_dict = {0: None, 1: np.byte, 2: np.int, 3: np.long, 4: np.float, 5: np.double, 6: np.complex, 7: np.str,
             8: object, 9: np.complex, 10: None, 11: object, 12: np.uint, 13: np.uint, 14: np.longlong, 15: np.uint}


def biginteger(x):
    return np.long(x)


def boolean(x):
    return np.bool(x)


def byte(x):
    return np.byte(x)


def double(x):
    return np.float(x)


def fix(x):
    return np.int(x)


def float(x):
    return np.float(x)


def long(x):
    return np.long(x)


def long64(x):
    return np.longlong(x)


def uint(x):
    return np.uint(x)


def ulong(x):
    return np.uint(x)


def ulong64(x):
    return np.uint(x)


"""
------------
"""


def poly(*args):
    """ polynom of args """
    return np.poly1d(*args)


def poly_2d(arr, p, q, *args, **kwargs):
    """ polynomial warping """
    raise NotImplementedError


def mean(arr, dim=None):
    """ mean of args """
    if dim is not None:
        dim -= 1
    return np.mean(arr, axis=dim)


def stddev(arr, dim=None):
    """ standard deviation """
    if dim is not None:
        dim -= 1
    return np.std(arr, axis=dim)


def variance(arr, dim=None):
    """ Variance """
    if dim is not None:
        dim -= 1
    return np.var(arr, axis=dim)


"""
------
"""


def convol(arr, kernel, *args, **kwargs):
    """ convolve arr with kernel """
    return np.convolve(arr, kernel)


"""
-----------------------------
    Array Initialization
-----------------------------
"""


def __array__(arr):
    """ make sure all arrays are numpy arrays """
    if not isinstance(arr, (list, tuple, np.ndarray)):
        arr = [arr]
    return np.array(arr)


def make_array(*n, increment=1, index=False, start=0, nozero=False, value=0, type=None, boolean=False, byte=False, complex=False, dcomplex=False, double=False, float=False, integer=False, l64=False, long=False, obj=False, ptr=False, string=False, uint=False, ul64=False, ulong=False):
    """ make array of given type and values """
    # determine type
    dtype = np.float
    if boolean is not None:
        dtype = np.bool
    if byte is not None:
        dtype = np.byte
    if complex is not None:
        dtype = np.complex
    if dcomplex is not None:
        dtype = np.complex
    if double is not None:
        dtype = np.double
    if float is not None:
        dtype = np.float
    if integer is not None:
        dtype = np.int
    if l64 is not None:
        dtype = np.longlong
    if long is not None:
        dtype = np.long
    if obj is not None:
        dtype = object
    if ptr is not None:
        dtype = None
    if string is not None:
        dtype = np.str
    if uint is not None:
        dtype = np.uint
    if ul64 is not None:
        dtype = np.uint
    if ulong is not None:
        dtype = np.uint

    if type is not None:
        dtype = type_dict[type]

    if nozero:
        return np.empty(n, dtype=dtype)
    if index:
        n = np.prod(n) * increment
        return np.arange(start, start + n, increment, dtype=dtype).reshape(n)
    return np.full(n, value, dtype=dtype)


def replicate(value, *n):
    """ create array with type of value and dimensions n """
    return np.full(n, value)


def identity(n, double=False):
    """ identity matrix with dimensions n """
    return np.identity(n)


def boolarr(*n):
    """ bool array """
    return intarr(*n, dtype=np.bool)


def bytarr(*n, nozero=False):
    """ byte array """
    return intarr(*n, nozero=nozero, dtype=np.byte)


def complexarr(*n, nozero=False):
    """ complex array """
    return intarr(*n, nozero=nozero, dtype=np.complex)


def dblarr(*n, nozero=False):
    """ double array """
    return intarr(*n, nozero=nozero, dtype=np.float)


def dcomplexarr(*n, nozero=False):
    """ complex double array """
    return intarr(*n, nozero=nozero, dtype=np.complex)


def fltarr(*n, nozero=False):
    """ float array """
    return intarr(*n, nozero=nozero, dtype=np.float)


def intarr(*n, nozero=False, dtype=np.int):
    """ integer array """
    if nozero:
        return np.empty(n, dtype=dtype)
    return np.zeros(n, dtype=dtype)


def l64arr(*n, nozero=False):
    """ longlong array """
    return intarr(*n, nozero=nozero, dtype=np.longlong)


def lonarr(*n, nozero=False):
    """ long array """
    return intarr(*n, nozero=nozero, dtype=np.long)


def objarr(*n):
    """ object array """
    return intarr(*n, dtype=object)


def ptrarr(*n):
    """ object array """
    return intarr(*n, dtype=object)


def strarr(*n):
    """ string array """
    return intarr(*n, dtype=np.str)


def uintarr(*n):
    """ uint array """
    return intarr(*n, dtype=np.uint)


def ulon64arr(*n):
    """ uint array """
    return intarr(*n, dtype=np.uint)


def ulonarr(*n):
    """ uint array """
    return intarr(*n, dtype=np.uint)


def bindgen(*args, increment=1, start=0):
    """ create byte array """
    return indgen(*args, increment=increment, start=start, dtype=np.byte)


def cindgen(*args, increment=1, start=0):
    """ complex array """
    return indgen(*args, increment=increment, start=start, dtype=np.complex)


def dcindgen(*args, increment=1, start=0):
    """ complex double array """
    return indgen(*args, increment=increment, start=start, dtype=np.complex)


def dindgen(*args, increment=1, start=0):
    """ return a double(float) array with size n """
    return indgen(*args, increment=increment, start=start, dtype=np.float64)


def findgen(*args, increment=1, start=0):
    """ float array """
    return indgen(*args, increment=increment, start=start, dtype=np.float)


def indgen(*args, increment=1, start=0, dtype=np.int):
    """
    return an integer array with size n,
    where each element is equal to its index + start
    """
    args = [int(a) for a in args]
    n = int(np.prod(args)) * increment
    return np.arange(start, start + n, increment, dtype=dtype).reshape(args)


def l64indgen(*args, increment=1, start=0):
    """ long array """
    return indgen(*args, increment=increment, start=start, dtype=np.longlong)


def lindgen(*args, increment=1, start=0):
    """ return a long array with size n """
    return indgen(*args, increment, start, dtype=np.long)


def sindgen(*args, increment=1, start=0):
    """ return a string array with size n """
    n = np.prod(args) * increment
    temp = [str(i) for i in range(start, start + n, increment)]
    return np.array(temp).reshape(args)


def uindgen(*args, increment=1, start=0):
    """ return a long array with size n """
    return indgen(*args, increment, start, dtype=np.uint)


def ul64indgen(*args, increment=1, start=0):
    """ return a long array with size n """
    return indgen(*args, increment, start, dtype=np.uint)


def ulindgen(*args, increment=1, start=0):
    """ return a long array with size n """
    return indgen(*args, increment, start, dtype=np.uint)


"""
---------------
    Strings
---------------
"""


def strcmp(s1, s2, n=None, fold_case=False):
    """ compare s1 and s2 """
    if n is not None:
        s1 = s1[:n]
        s2 = s2[:n]
    if fold_case:
        s1 = s1.casefold()
        s2 = s2.casefold()
    return s1 == s2


def strcompress(s, remove_all=False):
    """ compress all whitespace into one (or none) in s """
    s = s.split()
    if remove_all:
        sep = ''
    else:
        sep = ' '
    return strjoin(s, sep)


def stregex(s, pattern, **kwargs):
    """ match string to regex pattern """
    return re.search(pattern, s).start()


def string(s, *args, format=None):
    """ convert to string """
    return str(s)


def strjoin(s, delimiter='', single=False):
    """ Collapse string array into single line """
    s = (('%r' + delimiter) * len(s)) % tuple(s)
    return s[:-len(delimiter)]


def strlen(s):
    """ return length of string """
    return len(s)


def strlowcase(s):
    """ lower case string """
    return s.lower()


def strmatch(s, pattern, fold_case=False):
    """ compare string to pattern """
    if fold_case:
        s = s.casefold()
        pattern = pattern.casefold()
    if isinstance(s, str):
        return fnmatch.fnmatch(s, pattern)
    else:
        return fnmatch.filter(s, pattern)


def strmid(s, i, j=None, reverse=False):
    """ extract a substring """
    if reverse:
        i = -i
        if j is not None:
            j = -j - 2
    if j is None:
        return s[i:]
    return s[i:j + 1]


def strpos(s, sub, reverse_search=False):
    """ find index of sub in s"""
    if isinstance(s, (list, np.ndarray)):
        return np.array([strpos(i, sub, reverse_search=reverse_search) for i in s])
    if reverse_search:
        return s.rfind(sub)
    return s.find(sub)


def strput(dest, source, position=0):
    """ replace parts of dest with source at position """
    if position < 0:
        position = 0
    return dest[:position] + source + dest[position + len(source):]


def strsplit(s, pattern=' ', count=None, fold_case=False, extract=False):
    """ split string at any character of pattern """
    if fold_case:
        s = s.casefold()
        pattern = pattern.casefold()
    if not isinstance(s, list):
        s = [s]
    for c in pattern:
        st = []
        for s1 in s:
            st = st + s1.split(c)
        s = st

    return s


def strtrim(s, flag=0):
    """ strip whitespaces """
    if isinstance(s, list) or isinstance(s, np.ndarray):
        return np.array([strtrim(i, flag=0) for i in s])
    if not isinstance(s, str):
        s = str(s)
    if flag == 0:
        return s.rstrip()
    if flag == 1:
        return s.lstrip()
    else:
        return s.strip()


def strupcase(s):
    """ upper case string """
    return s.upper()


"""
------------
    Fits
------------
"""


def poly_fit(*args):
    # TODO check this works as expected in idl
    return scipy.optimize.curve_fit(*args)


def curvefit(*args, **kwargs):
    raise NotImplementedError


"""
------------
    Plot
------------
"""
# TODO all of plotting
# TODO should be none blocking
plt.ion()


def loadct(*args, **kwargs):
    """ Obsolete """
    return


def plot(*args, **kwargs):
    """ Plot y against x, creates a new canvas """
    plt.figure()
    plt.plot(*args)


def oplot(*args, **kwargs):
    """ Overplot, use existing canvas """
    plt.plot(*args)


def display(*args, **kwargs):
    """ Display a 2D image """
    plt.implot(*args)
    return args


def histogram(arr, binsize=1, min=None, max=None, nbins=None):
    raise NotImplementedError


def hist_equal(arr, *args, **kwargs):
    raise NotImplementedError


def tvlct(*args):
    """ something about colors """
    pass


def device(filename=None, decomposed=None, **kwargs):
    """ does something """
    return None


"""
-------------
    Switch   
-------------
"""

# This class provides the functionality we want. You only need to look at
# this if you want to know how this works. It only needs to be defined
# once, no need to muck around with its internals.
# Some modifications to make it work for case blocks as well


class switch(object):
    def __init__(self, value, fallthrough=True):
        self.value = value
        self.fall = False
        self.fallthrough = fallthrough
        self.foundvalue = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration

    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args and not self.foundvalue:
            return True
        elif self.value in args:  # changed for v1.5, see below
            if self.fallthrough:
                self.fall = True
            else:
                self.foundvalue = True
            return True
        else:
            return False


"""
---------------------
    Common Table
---------------------
"""


def common(table, n):
    """ IDL common table """
    # n = number of parameters to read
    # TODO better implementation
    table = table + '.txt'
    res = [[] for i in range(n)]  # initialize output
    with open(table, 'r') as file:
        for line in file.readlines():
            # count whitespaces
            i = 0
            while line[0] == ' ':
                i += 1
                line = line[1:]

            j = i // 3
            res[j].append(line.strip())
            k = [i for i in range(j)]
            for m in k:
                if len(res[j]) > len(res[m]):
                    res[m].append(res[m][-1])

    return [np.array(r) for r in res]


"""
-----------------
    Dialog
-----------------
"""


def dialog_pickfile(*args, **kwargs):
    return ''


"""
-----
https://github.com/mperrin/pyidlastro/blob/master/pyidlastro/idl_language.py
-----
"""

# from http://www.scipy.org/Cookbook/Rebinning, modified slightly


def rebin_avg(a, *newshape0):
    # NOTE: does not swap dimensions for IDL/Py style
    '''rebin ndarray data into a smaller ndarray of the same rank whose dimensions
    are factors of the original dimensions. eg. An array with 6 columns and 4 rows
    can be reduced to have 6,3,2 or 1 columns and 4,2 or 1 rows.
    example usages:
    >>> a=rand(6,4); b=rebin(a,3,2)
    >>> a=rand(6); b=rebin(a,2)
    '''
    # swap axes order to allow IDL convention in the newshape0 argument
    newshape = newshape0[::-1]

    shape = a.shape
    lenShape = len(shape)
    factor = np.asarray(shape) / np.asarray(newshape)
    evList = ['a.reshape('] + \
             ['args[%d],factor[%d],' % (i, i) for i in range(lenShape)] + \
             [')'] + ['.sum(%d)' % (i + 1) for i in range(lenShape)] + \
             ['/factor[%d]' % i for i in range(lenShape)]
    print(''.join(evList))
    return eval(''.join(evList))


def shift(array, s1, s2=0, s3=0):
    """
    Shift array by s1, values wrap around
    """
    dims = array.ndim
    # IDL/Py axes order swappage
    if dims == 1:
        return np.roll(array, s1)
    elif dims == 2:
        return np.roll(np.roll(array, s1, axis=1), s2, axis=0)
    elif dims == 3:
        return np.roll(np.roll(np.roll(array, s1, axis=2), s2, axis=1), s3, axis=0)
    else:
        print("Don't know how to shift > 3D arrays...")
    return array

# from http://www.scipy.org/Cookbook/Rebinning, modified slightly


def rebin(a, *newshape0):
    '''
    Rebin an array to a new shape.
    Arbitrary new shape allowed, no interpolation done.
    '''
    # swap axes order to allow IDL convention in the newshape0 argument
    newshape = newshape0[::-1]

    assert len(a.shape) == len(newshape)

    slices = [slice(0, old, float(old) / new)
              for old, new in zip(a.shape, newshape)]
    coordinates = np.mgrid[slices]
    # choose the biggest smaller integer index
    indices = coordinates.astype('i')
    return a[tuple(indices)]

# from http://www.scipy.org/Cookbook/Rebinning


def congrid(a, newdims, method='linear', centre=False, minusone=False):
    '''Arbitrary resampling of source array to new dimension sizes.
    Currently only supports maintaining the same number of dimensions.
    To use 1-D arrays, first promote them to shape (x,1).

    Uses the same parameters and creates the same co-ordinate lookup points
    as IDL''s congrid routine, which apparently originally came from a VAX/VMS
    routine of the same name.
    method:
    neighbour - closest value from original data
    nearest and linear - uses n x 1-D interpolations using
                         scipy.interpolate.interp1d
    (see Numerical Recipes for validity of use of n 1-D interpolations)
    spline - uses ndimage.map_coordinates
    centre:
    True - interpolation points are at the centres of the bins
    False - points are at the front edge of the bin
    minusone:
    For example- inarray.shape = (i,j) & new dimensions = (x,y)
    False - inarray is resampled by factors of (i/x) * (j/y)
    True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
    This prevents extrapolation one element beyond bounds of input array.
    '''
    if not a.dtype in [n.float64, n.float32]:
        a = np.cast[float](a)

    m1 = np.cast[int](minusone)
    ofs = np.cast[int](centre) * 0.5
    old = np.array(a.shape)
    ndims = len(a.shape)
    if len(newdims) != ndims:
        print("[congrid] dimensions error. "
              "This routine currently only support "
              "rebinning to the same number of dimensions.")
        return None
    newdims = np.asarray(newdims, dtype=float)
    dimlist = []

    if method == 'neighbour':
        for i in range(ndims):
            base = np.indices(newdims)[i]
            dimlist.append((old[i] - m1) / (newdims[i] - m1)
                           * (base + ofs) - ofs)
        cd = np.array(dimlist).round().astype(int)
        newa = a[list(cd)]
        return newa

    elif method in ['nearest', 'linear']:
        # calculate new dims
        for i in range(ndims):
            base = np.arange(newdims[i])
            dimlist.append((old[i] - m1) / (newdims[i] - m1)
                           * (base + ofs) - ofs)
        # specify old dims
        olddims = [np.arange(i, dtype=np.float) for i in list(a.shape)]

        # first interpolation - for ndims = any
        mint = scipy.interpolate.interp1d(olddims[-1], a, kind=method)
        newa = mint(dimlist[-1])

        trorder = [ndims - 1] + range(ndims - 1)
        for i in range(ndims - 2, -1, -1):
            newa = newa.transpose(trorder)

            mint = scipy.interpolate.interp1d(olddims[i], newa, kind=method)
            newa = mint(dimlist[i])

        if ndims > 1:
            # need one more transpose to return to original dimensions
            newa = newa.transpose(trorder)

        return newa
    elif method in ['spline']:
        oslices = [slice(0, j) for j in old]
        oldcoords = np.ogrid[oslices]
        nslices = [slice(0, j) for j in list(newdims)]
        newcoords = np.mgrid[nslices]

        newcoords_dims = range(np.rank(newcoords))
        # make first index last
        newcoords_dims.append(newcoords_dims.pop(0))
        newcoords_tr = newcoords.transpose(newcoords_dims)
        # makes a view that affects newcoords

        newcoords_tr += ofs

        deltas = (np.asarray(old) - m1) / (newdims - m1)
        newcoords_tr *= deltas

        newcoords_tr -= ofs

        newa = scipy.ndimage.map_coordinates(a, newcoords)
        return newa
    else:
        print("Congrid error: Unrecognized interpolation type.\n",
              "Currently only \'neighbour\', \'nearest\',\'linear\',",
              "and \'spline\' are supported.")
        return None


def bytscl(array, max=None, min=None, nan=0, top=255):
    """
    see http://star.pst.qub.ac.uk/idl/BYTSCL.html
    note that IDL uses slightly different formulae for bytscaling floats and ints.
    here we apply only the FLOAT formula...
    """
    if max is None:
        max = np.nanmax(array)
    if min is None:
        min = np.nanmin(array)

    # return (top+0.9999)*(array-min)/(max-min)
    return np.maximum(np.minimum(((top + 0.9999) * (array - min) / (max - min)).astype(np.int16), top), 0)
