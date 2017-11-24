"""
Solve the linearized minimization Problem Phi = sum(G*P - F) + lam * R
"""

import numpy as np
from scipy.linalg import solve_banded
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.optimize import curve_fit, fsolve


class solution:
    """ Wrapper class for the functions """

    def __init__(self, dtype=np.float):
        self.dtype = dtype

    # TODO try to find best value for lambda
    # Brute Force Solution: try different values for lambda and find the best
    # What is the best lambda ??? Which metric is used to determine that?

    def Franklin(self, wl, f, g, lamb):
        """
        Solve the mimimazation problem to find the planetary spectrum
        F*x - G = 0
        wl_grid: Wavelength scale
        F: intermediary product F
        G: intermediary product G
        lam: regularization parameter lambda
        """
        # http://epubs.siam.org/doi/pdf/10.1137/0509044
        if isinstance(lamb, np.ndarray) and len(lamb) == 1:
            lamb = lamb[0]

        if isinstance(lamb, (int, float, np.int64, np.float)):
            lamb = np.full(len(wl), lamb, dtype=self.dtype)
        a, c = np.zeros(len(wl), dtype=np.float32), np.zeros(
            len(wl), dtype=self.dtype)
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

    def Tikhonov(self, wl, f, g, l):
        """ Using Sparse Matrixes, experimental might be faster? """
        """ Also Tikhonov regularization instead of Franklin """

        b = np.sum(f, axis=0)
        r = np.sum(g, axis=0)

        n = len(wl)
        # Difference Operator D
        D = self.__difference_matrix__(n)

        A = diags(b, 0)
        # Inverse
        A.I = diags(1 / b, 0)

        return spsolve(A + l**2 * A.I * D.T * D, r)

    def __difference_matrix__(self, size):
        a = c = np.full(size - 1, -1)
        b = np.full(size, 2)
        b[0] = b[-1] = 1
        return diags([a, b, c], offsets=[-1, 0, 1])

    def best_lambda(self, wl, f, g, sample_range=[1e-5, 1e3], npoints=100, method='Franklin'):
        """ Use the L-curve algorithm to find the best regularization parameter lambda """
        # http://www2.compute.dtu.dk/~pcha/DIP/chap5.pdf
        # TODO: is there a good sample range for all situations?
        # TODO: Maybe an iterative approach works better/faster?
        b = np.sum(f, axis=0)
        r = np.sum(g, axis=0)

        D = self.__difference_matrix__(len(wl))
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

        plot = False
        sampling = 'log'
        if sampling == 'log':
            sample_range = np.log(sample_range)
            lamb = np.logspace(sample_range[0], sample_range[1], npoints)
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

                if abs(1 - p[0] / x_min) < abs(1 - p[1]/ y_min):
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
        y_scale = y.min()**-1
        x_scale = x.min()**-1
        d = distance(x * x_scale, y * y_scale)

        if plot:
            import matplotlib.pyplot as plt
            plt.plot(x, y, '+')
            plt.plot(p1[0], p1[1], 'r+')
            plt.plot(p2[0], p2[1], 'g+')
            plt.plot(x[np.argmin(d)], y[np.argmin(d)], 'd')
            plt.show()

        return lamb[np.argmin(d)]

    def best_lambda_dirty(self, wl, f, g, lamb0=100):
        """ Using dirty limitation of max(solution) == 1 """
        def func(x): return self.solve(wl, f, g, np.abs(x)).max() - 1
        lamb, _, _, _ = fsolve(func, x0=lamb0, full_output=True)
        lamb = np.abs(lamb[0])
        return lamb
