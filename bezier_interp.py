from idl_lib import *
from idl_lib import __array__
import _global


def bezier_interp(xa, ya, y2a, x, double=None):
    """
    # Performs cubic Bezier spline interpolation
    # IMPORTANT: the XA array must be monotonic!!!
    """
    n = n_elements(xa)
    m = n_elements(x)
    if keyword_set(double):
        y = dblarr(m)
    else:
        y = x * 0

    _, nii = where(x >= min(xa) and x <= max(xa), count='nii')

    if nii == 0:
        return y

    klo = ((value_locate(xa, x) < (n - 2))) > 0
    khi = klo + 1
    if keyword_set(double):
        h = double(xa[khi]) - double(xa[klo])
        y1 = double(ya[klo])
        y2 = double(ya[khi])
    else:
        h = xa[khi] - xa[klo]
        y1 = ya[klo]
        y2 = ya[khi]
    a = (xa[khi] - x) / h
    b = (x - xa[klo]) / h
    c0 = y1 + h / 3e0 * y2a[klo]
    c1 = y2 - h / 3e0 * y2a[khi]
    y = a * a * a * y1 + 3e0 * a * a * b * c0 + \
        3e0 * a * b * b * c1 + b * b * b * y2

    return y
