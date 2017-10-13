from idl_lib import *
from idl_lib import __array__
import _global


def bezier_init(x, y, double=None):
    """
    # Computes automatic control points for cubic Bezier splines
    # IMPORTANT: the X array must be monotonic!!!
    #
    # If we define for points x_a and x_b along a ray:
    # u = (x - x_a)/(x_b - x_a)
    # then any function can be fit with a Bezier spline as
    # f(u) = f(x_a)*(1 - u)^3 + 3*C0*u*(1-u)^2 + 3*C1*u^2*(1-u) + f(x_b)*u^3
    # where C0 and C1 are the local control parameters.
    #
    # Control parameter1 for interval [x_a, x_b] are computed as:
    # C0 = f(x_a) + delta/3*D'_a
    # and
    # C1 = f(x_b) - delta/3*D'_b
    #
    # If D(b-1/2)*D(b+1/2) > 0 then
    # D'_b  = D(b-1/2)*D(b+1/2) / (alpha*D(b+1/2) + (1-alpha)*D(b-1/2))
    # Else
    # D'_b  = 0
    #
    # D(b-1/2) = [f(x_b) - f(x_a)] / delta
    # D(b+1/2) = [f(x_c) - f(x_b)] / delta'
    # alpha    = [1 + delta'/(delta + delta')]/3
    # delta    = x_b - x_a
    # delta'   = x_c - x_b
    #
    # For the first and the last step we assume D(b-1/2)=D(b+1/2) and, therefore,
    # D'_b = D(b+1/2) for the first point and
    # D'_b = D(b-1/2) for the last point
    #
    # The actual interpolation is split in two parts. This INIT subroutine
    # computes the array if D'_b
    """

    n = n_elements(x)
    if (keyword_set(double)):
        y2 = dblarr(n)
        h2 = double(x[1] - x[0])
        der2 = double(y[1] - y[0]) / h2
        y2[0] = der2
        for i in np.arange(1, n - 2 + 1, 1):
            h1 = h2
            der1 = der2
            h2 = double(x[i + 1] - x[i])
            der2 = double(y[i + 1] - y[i]) / h2
            alpha = (1e0 + h2 / (h1 + h2)) / 3e0
            if (der1 * der2 > 0e0):
                y2[i] = der1 * der2 / (alpha * der2 + (1e0 - alpha) * der1)
            else:
                y2[i] = 0e0
        y2[n - 1] = der2
    else:
        y2 = x
        h2 = x[1] - x[0]
        der2 = (y[1] - y[0]) / h2
        y2[0] = der2
        for i in np.arange(1, n - 2 + 1, 1):
            h1 = h2
            der1 = der2
            h2 = x[i + 1] - x[i]
            der2 = (y[i + 1] - y[i]) / h2
            alpha = (1e0 + h2 / (h1 + h2)) / 3.
            if (der1 * der2 > 0e0):
                y2[i] = der1 * der2 / (alpha * der2 + (1. - alpha) * der1)
            else:
                y2[i] = 0.
        y2[n - 1] = der2
    return y2
