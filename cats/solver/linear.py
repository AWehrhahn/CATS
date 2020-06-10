"""
Solve the linearized minimization Problem Phi = sum(G*P - F) + lam * R
"""

import logging
import numpy as np
from numpy.linalg import norm
from scipy.linalg import solve_banded, dft
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.optimize import fsolve, minimize_scalar
from astropy.constants import c
from astropy import units as u
from tqdm import tqdm

from .solver import SolverBase
from ..reference_frame import TelescopeFrame, PlanetFrame
from ..spectrum import Spectrum1D


class LinearSolver(SolverBase):
    def __init__(
        self,
        detector,
        star,
        planet,
        method="Tikhonov",
        regularization=True,
        regularization_ratio=50,
        regularization_weight=None,
        plot=False,
    ):
        super().__init__(detector, star, planet)
        self.method = method
        self.regularization = regularization
        self.regularization_ratio = regularization_ratio
        self.regularization_weight = regularization_weight
        self.plot = plot
        self.difference_accuracy = 8

    @property
    def method(self):
        if self._method == "Tikhonov":
            return self.Tikhonov
        elif self._method == "Franklin":
            return self.Franklin
        else:
            return self._method

    @method.setter
    def method(self, value):
        self._method = value

    def _Franklin(self, A, D, l, g):
        return spsolve(A + l * D, g)

    def Franklin(self, f, g, lamb, spacing=None):
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
            lamb = np.full(len(f), lamb)
        a, c = np.zeros(len(f)), np.zeros(len(f))
        a[1:] = -lamb[:-1]
        c[:-1] = -lamb[1:]

        f[:-1] += lamb[:-1]
        f[1:] += lamb[1:]

        ab = np.array([a, f, c])
        return solve_banded((1, 1), ab, g)

    def _Tikhonov(self, A, D, l, g):
        return spsolve(A ** 2 + l ** 2 * D.T * D, A * g)

    def Tikhonov(self, f, g, l, spacing=None):
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

        f = np.asarray(f)
        g = np.asarray(g)

        # Create the matrix A (and its inverse)
        if spacing is None:
            n = np.arange(len(f))
            D = self.gradient_matrix(n)
        else:
            D = self.gradient_matrix(spacing)
        A = diags(f, 0)

        # Solve the equation
        sol = self._Tikhonov(A, D, l, g)
        return sol

    def difference_matrix_2(self, spacing):
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
        size = spacing.size
        h = np.diff(spacing)
        a = c = -1 / h
        b = np.zeros(size)
        b[1:-1] = (h[1:] + h[:-1]) / (h[1:] * h[:-1])
        b[0] = 1 / h[0]
        b[-1] = 1 / h[-1]
        return diags([a, b, c], offsets=[-1, 0, 1])

    def gradient_matrix(self, grid):
        size = len(grid)
        if self.difference_accuracy == 2:
            factors = [-1 / 2, 0.0, 1 / 2]
        elif self.difference_accuracy == 4:
            factors = [1 / 12, -2 / 3, 0.0, 2 / 3, -1 / 12]
        elif self.difference_accuracy == 6:
            factors = [-1 / 60, 3 / 20, -3 / 4, 0.0, 3 / 4, -3 / 20, 1 / 60]
        elif self.difference_accuracy == 8:
            factors = [
                1 / 280,
                -4 / 105,
                1 / 5,
                -4 / 5,
                0.0,
                4 / 5,
                -1 / 5,
                4 / 105,
                -1 / 280,
            ]
        else:
            raise ValueError

        nf = len(factors)
        offsets = np.arange(nf) - nf // 2
        columns = [np.full(size - abs(j), f) for j, f in zip(offsets, factors)]
        grad = diags(columns, offsets=offsets)
        return grad

    def fourier_matrix(self, size):
        return dft(size, "sqrtn")

    def best_lambda_dirty(self, wl, f, g, lamb0=100):
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
            return self.Franklin(wl, f, g, np.abs(x)).max() - 1

        lamb, _, _, _ = fsolve(func, x0=lamb0, full_output=True)
        lamb = np.abs(lamb[0])
        return lamb

    def best_lambda(self, f, g, spacing=None):
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
            how much more important y is relative to x (default is 50, which gives "nice" results)
        method: {'Tikhonov', 'Franklin'}, optional
            which regularization method to use, for best lambda finding, should be the same, that is used for the final calculation (default is 'Tikhonov')
        plot : bool, optional
            show a plot of the L curve if True (the default is False, which means no plot)

        Returns
        -------
        lambda : float
            Best fit regularization parameter lambda
        """

        def get_point(lamb, A, D, r):
            """ calculate points of the L-curve"""
            if self._method == "Tikhonov":
                sol = self._Tikhonov(A, D, lamb, r)
            if self._method == "Franklin":
                sol = spsolve(A + lamb * D, r)

            x = norm(A * sol - r, 2)
            y = norm(D * sol, 2)
            return x, y

        def func(lamb, ratio, A, D, r, angle=-np.pi / 4):
            """ get "goodness" value for a given lambda using L-parameter """
            progress.update(1)
            x, y = get_point(lamb, A, D, r)
            # scale and rotate point
            return -x * np.sin(angle) + y * ratio * np.cos(angle)

        # reduce data, and filter nans
        b, r = f, g
        mask = np.isfinite(b) & np.isfinite(r)
        b, r = b[mask], r[mask]

        # prepare matrices
        if spacing is not None:
            D = self.gradient_matrix(spacing)
        else:
            spacing = np.arange(len(b))
            D = self.gradient_matrix(spacing)

        # D = __fourier_matrix__(len(b))
        A = diags(b, offsets=0)
        A.I = diags(1 / b, 0)

        # Calculate best lambda
        progress = tqdm(total=500)
        ratio = self.regularization_ratio
        res = minimize_scalar(func, args=(ratio, A, D, r), method="brent")
        progress.close()
        if self.plot:
            import matplotlib.pyplot as plt

            ls = np.geomspace(1, 1e6, 300)
            tmp = [get_point(l, A, D, r) for l in ls]
            x = np.array([t[0] for t in tmp])
            y = np.array([t[1] for t in tmp])

            p1 = get_point(10, A, D, r)
            p2 = get_point(1e6, A, D, r)
            p3 = get_point(res.x, A, D, r)

            def rotate(x, y, angle):
                i = x * np.cos(angle) + y * ratio * np.sin(angle)
                j = -x * np.sin(angle) + y * ratio * np.cos(angle)
                return i, j

            angle = -np.pi / 4
            x, y = rotate(x, y, angle)
            p1 = rotate(*p1, angle)
            p2 = rotate(*p2, angle)
            p3 = rotate(*p3, angle)

            plt.plot(x, y, "+")
            plt.plot(p1[0], p1[1], "r+")
            plt.plot(p2[0], p2[1], "g+")
            plt.plot(p3[0], p3[1], "d")
            plt.loglog()
            plt.xlabel(r"$||\mathrm{Residual}||_2$")
            plt.ylabel(str(ratio) + r"$ * ||\mathrm{first derivative}||_2$")
            plt.show()

        return res.x

    def solve(self, times, wavelength, spectra, stellar, intensities, telluric):
        """
        Find the least-squares solution to the linear equation
        f * x - g = 0
        """
        wave, f, g = self.prepare_fg(
            times, wavelength, spectra, stellar, intensities, telluric
        )

        if self.regularization:
            if self.regularization_weight is None:
                # regweight = 200
                regweight = self.best_lambda(f, g)
                print("Regularization weight: ", regweight)
            else:
                regweight = self.regularization_weight
        else:
            regweight = 0

        x0 = self.Tikhonov(f, g, regweight)

        # Normalize x0, each segment individually
        diff = np.diff(wave)
        idx = [0, *np.where(diff > 1000 * np.median(diff))[0], -1]

        for left, right in zip(idx[:-1], idx[1:]):
            x0[left:right] -= np.min(x0[left:right])
            x0[left:right] /= np.nanpercentile(x0[left:right], 90)

        spec = Spectrum1D(
            flux=x0 << u.one,
            spectral_axis=wave << u.AA,
            source="Linear solver",
            description="recovered planet transmission spectrum",
            reference_frame="planet",
            star=self.star,
            planet=self.planet,
            observatory_location=self.detector.observatory,
        )

        return spec
