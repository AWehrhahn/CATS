"""
Solve the linearized minimization Problem Phi = sum(G*P - F) + lam * R
"""

import numpy as np
from numpy.linalg import norm
from scipy.linalg import solve_banded
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.optimize import fsolve, minimize_scalar

from log import log

def Franklin(wl, f, g, lamb):
    """Solve the minimization problem f * x - g = 0

    Use simple Franklin regularization with parameter lamb
    http: epubs.siam.org/doi/pdf/10.1137/0509044

    Parameters:
    ----------
    wl : np.ndarray
        wavelength grid
    f : np.ndarray
        f
    g : np.ndarray
        g
    lamb : float
        regularization parameter
    Returns
    -------
    np.ndarray
        The planetary spectrum
    """
    #
    if isinstance(lamb, np.ndarray) and len(lamb) == 1:
        lamb = lamb[0]

    if isinstance(lamb, (int, float, np.int64, np.float)):
        lamb = np.full(len(wl), lamb)
    a, c = np.zeros(len(wl)), np.zeros(len(wl))
    a[1:] = -lamb[:-1]
    c[:-1] = -lamb[1:]

    b = np.sum(f, axis=0)
    r = np.sum(g, axis=0)
    b[:-1] += lamb[:-1]
    b[1:] += lamb[1:]

    ab = np.array([a, b, c])
    # func = np.sum((so / ff - sigma_p / sigma_a * ke + ke *
    #               (np.tile(planet, n_phase).reshape((n_phase, len(planet)))) - obs / ff)**2)
    #reg = lamb * np.sum((sol[1:] - sol[:-1])**2)
    return solve_banded((1, 1), ab, r)


def Tikhonov(f, g, l):
    """Solve f * x = g, with Tikhonov regularization parameter l

    Solve the equation diag(f) + l**2 * diag(f).I * D.T * D = g
    where D is the difference operator matrix, and diag the diagonal matrix

    Parameters:
    ----------
    f : np.ndarray
    g : np.ndarray
    l : float
        Tikhonov regularization parameter
    Returns
    -------
    np.ndarray
        x
    """
    b = np.sum(f, axis=0)
    r = np.sum(g, axis=0)

    mask = (~np.isnan(b)) & (~np.isnan(r))
    b = b[mask]
    r = r[mask]

    n = len(b)
    # Difference Operator D
    D = __difference_matrix__(n)

    A = diags(b, 0)
    # Inverse
    A.I = diags(1 / b, 0)

    sol = spsolve(A + l**2 * A.I * D.T * D, r)

    # TODO use nan instead of 0
    s = np.full(len(mask), 0, dtype=float)
    s[mask] = sol
    return s


def __difference_matrix__(size):
    """Get the difference operator matrix

    The difference operator is a matrix with the diagonal = 2, and both first offsets = -1

    Parameters:
    ----------
    size : int
        the size of the returned matrix
    Returns
    -------
    dense matrix
        the difference matrix of size size
    """
    a = c = np.full(size - 1, -1)
    b = np.full(size, 2)
    b[0] = b[-1] = 1
    return diags([a, b, c], offsets=[-1, 0, 1])


def best_lambda(f, g, ratio=40, method='Tikhonov', plot=False):
    """Use the L-curve algorithm to find the best regularization parameter lambda

    http://www2.compute.dtu.dk/~pcha/DIP/chap5.pdf
    
    x = Residual
    y = First Derivative
    
    this will create a L shaped curve, with the optimal lambda in the corner
    to find that value rotate the curve by pi/4 (45 degrees) and search for the minimum

    Parameters:
    ----------
    f : np.ndarray
    g : np.ndarray
    ratio: float, optional
        how much more important y is relative to x (default is 80, which gives "nice" results)
    method: {'Tikhonov', 'Franklin'}, optional
        which regularization method to use, for best lambda finding, should be the same, that is used for the final calculation (default is 'Tikhonov')
    plot : bool, optional
        show a plot of the L curve if True (the default is False, which means no plot)

    Returns
    -------
    lambda : float
        Best fit regularization parameter lambda
    """

    log(2, 'DeltaX/Residual ratio:', ratio)

    def get_point(lamb, A, D, r):
        """ calculate points of the L-curve"""
        if method == 'Tikhonov':
            sol = spsolve(A + lamb**2 * A.I * D.T * D, r)
        if method == 'Franklin':
            sol = spsolve(A + lamb * D, r)

        x = norm(A * sol - r, 2)
        y = norm(D * sol, 2)
        return x, y

    def func(lamb, ratio, A, D, r, angle=-np.pi / 4):
        """ get "goodness" value for a given lambda using L-parameter """
        x, y = get_point(lamb, A, D, r)
        # scale and rotate point
        return -x * np.sin(angle) + y * ratio * np.cos(angle)

    # reduce data, and filter nans
    b, r = np.sum(f, axis=0), np.sum(g, axis=0)
    mask = ~np.isnan(b) & ~np.isnan(r)
    b, r = b[mask], r[mask]

    # prepare matrices
    D = __difference_matrix__(len(b))
    A = diags(b, offsets=0)
    A.I = diags(1 / b, 0)

    # Calculate best lambda
    res = minimize_scalar(func, args=(ratio, A, D, r))

    if plot:
        import matplotlib.pyplot as plt
        ls = np.geomspace(1, 1e6, 300)
        tmp = [get_point(l, A, D, r) for l in ls]
        x = np.array([t[0] for t in tmp])
        y = np.array([t[1] for t in tmp])

        p1 = get_point(10, A, D, r)
        p2 = get_point(1e6, A, D, r)
        p3 = get_point(res.x, A, D, r)

        def rotate(x, y, angle):
            i = x * np.cos(angle) + y *ratio* np.sin(angle)
            j = -x * np.sin(angle) + y *ratio* np.cos(angle)
            return i, j

        angle = -np.pi/4
        x, y = rotate(x, y, angle)
        p1 = rotate(*p1, angle)
        p2 = rotate(*p2, angle)
        p3 = rotate(*p3, angle)

        plt.plot(x, y, '+')
        plt.plot(p1[0], p1[1], 'r+')
        plt.plot(p2[0], p2[1], 'g+')
        plt.plot(p3[0], p3[1], 'd')
        plt.loglog()
        plt.xlabel(r'$||\mathrm{Residual}||_2$')
        plt.ylabel(str(ratio) + r'$ * ||\mathrm{first derivative}||_2$')
        plt.show()

    #TODO scale with atmosphere height?????
    return res.x


def best_lambda_dirty(wl, f, g, lamb0=100):
    """ Use a simple hack to get a good approximation of the best lambda

    This just tests for which regularization lambda, the maximum value is 1
    The linear equation to solve is f * x - g = 0

    This is a dirty hack that has no reason to work as well as it does,
    but it gets good results and is about 2 Orders of magnitudes
    faster than the regular best_lambda function

    Parameters:
    ----------
    wl : np.ndarray
        wavelength grid
    f : np.ndarray
    g : np.ndarray
    lamb0 : float, optional
        The starting regularization parameter (the default is 100)

    Returns
    -------
    lambda: float
        Best fit regularization parameter lambda, hopefully
    """

    def func(x):
        return Franklin(wl, f, g, np.abs(x)).max() - 1
    lamb, _, _, _ = fsolve(func, x0=lamb0, full_output=True)
    lamb = np.abs(lamb[0])
    return lamb
