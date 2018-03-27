"""
Solve the linearized minimization Problem Phi = sum(G*P - F) + lam * R
"""

import numpy as np
from scipy.linalg import solve_banded
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.optimize import fsolve


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

    mask = ~np.isnan(b)
    b = b[mask]
    r = r[mask]

    n = len(b)
    # Difference Operator D
    D = __difference_matrix__(n)

    A = diags(b, 0)
    # Inverse
    A.I = diags(1 / b, 0)

    sol = spsolve(A + l**2 * A.I * D.T * D, r)

    #TODO use nan instead of 0
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


def RidgeRegression(f, g, l):
    """Use Ridge Regression to solve f*x-g=0

    .. note:: WARNING: EXPERIMENTAL

    Parameters:
    ----------
    f : np.ndarray
        f
    g : np.ndarray
        g
    l : float
        regulaization paramter
    Returns
    -------
    np.ndarray
        Planet spectrum
    """

    import sklearn.linear_model as linear

    b = np.sum(f, axis=0)
    r = np.sum(g, axis=0)

    A = diags(b, 0)

    rcv = linear.RidgeCV(fit_intercept=False, cv=10, alphas=(l,))
    rcv.fit(A, r)
    return rcv.predict(A)


def best_lambda(wl, f, g, sample_range=[1e-4, 1e6], npoints=300, method='Franklin', plot=False, sampling='log'):
    """Use the L-curve algorithm to find the best regularization parameter lambda

    http://www2.compute.dtu.dk/~pcha/DIP/chap5.pdf
    TODO: is there a good sample range for all situations?
    TODO: Maybe an iterative approach works better/faster?

    Parameters:
    ----------
    wl : np.ndarray
        wavelength grid
    f : np.ndarray
    g : np.ndarray
    sample_range : tuple(int), optional
        range of lambda values to test (the default is (1e-4, 1e6), which should be enough)
    npoints : int, optional
        number of sampling points (the default is 300, which should be enough)
    plot : bool, optional
        show a plot of the L curve if True (the default is False, which means no plot)

    Returns
    -------
    lambda : float
        Best fit regularization parameter lambda
    """
    b = np.sum(f, axis=0)
    r = np.sum(g, axis=0)

    D = __difference_matrix__(len(wl))
    A = diags(b, offsets=0)
    A.I = diags(1 / b, 0)

    def get_point(lamb):
        """ calculate points of the L-curve"""
        if method == 'Tikhonov':
            #    sol = self.Tikhonov(wl, f, g, lamb)
            sol = spsolve(A + lamb**2 * A.I * D.T * D, r)
        if method == 'Franklin':
            #    sol = self.Franklin(wl, f, g, lamb)
            sol = spsolve(A + lamb * D, r)
        y = np.sum((D * sol)**2)
        x = np.sum(((A + lamb * D) * sol - r)**2)
        return x, y

    sampling = 'log'
    if sampling == 'log':
        lamb = np.geomspace(sample_range[0], sample_range[1], npoints)
        tmp = [get_point(l) for l in lamb]
        x = np.array([t[0] for t in tmp])
        y = np.array([t[1] for t in tmp])
        p1 = [x[0], y[0]]
        p2 = [x[-1], y[-1]]

    if sampling == 'iterative':
        # TODO doesn't work properly
        _lamb = np.empty(npoints)
        _x = np.empty(npoints)
        _y = np.empty(npoints)

        l_low = sample_range[0]
        l_high = sample_range[1]
        lamb = (l_low + l_high) / 2

        p1 = get_point(l_low)
        p2 = get_point(l_high)

        _lamb[0] = l_low
        _lamb[1] = l_high
        x_min = p1[0]
        y_min = p2[1]

        _x[0] = p1[0]
        _x[1] = p2[0]
        _y[0] = p1[1]
        _y[1] = p2[1]

        for i in range(2, npoints):
            p = get_point(lamb)
            _lamb[i] = lamb
            _x[i] = p[0]
            _y[i] = p[1]

            if abs(1 - p[0] / x_min) < abs(1 - p[1] / y_min):
                l_low = lamb
                lamb = (lamb + l_high) / 2
            else:
                l_high = lamb
                lamb = (lamb + l_low) / 2

        sort = np.argsort(_x)
        x = _x[sort]
        y = _y[sort]
        lamb = _lamb[sort]

    def distance(x, y):
        return x**2 + y**2

    # Scales are necessary as large difference in size will make x and y incomparable
    # TODO Standardize x, y instead, i.e. sum(x**2) = 1, sum will be dominated by largest value
    y_scale = np.max(y)**-1
    x_scale = np.max(x)**-1
    d = distance(x * x_scale, y * y_scale)

    if plot:
        import matplotlib.pyplot as plt
        plt.plot(x * x_scale, y * y_scale, '+')
        plt.plot(p1[0] * x_scale, p1[1] * y_scale, 'r+')
        plt.plot(p2[0] * x_scale, p2[1] * y_scale, 'g+')
        plt.plot(x[np.argmin(d) - 10] * x_scale,
                 y[np.argmin(d) - 10] * y_scale, 'd')
        plt.show()

    return lamb[np.argmin(d) - 10]


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
