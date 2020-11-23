"""
Solve the linearized minimization Problem Phi = sum(G*P - F) + lam * R
"""

import logging
import numpy as np
from numpy.linalg import norm
from scipy.linalg import solve_banded, dft
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.optimize import fsolve, minimize_scalar, minimize
from scipy.special import binom
from scipy.sparse import csc_matrix, lil_matrix
from astropy.constants import c
from astropy import units as u
from tqdm import tqdm
import matplotlib.pyplot as plt

from .solver import SolverBase
from ..least_squares.least_squares import least_squares

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
        self.difference_accuracy = 2
        self.normalize = False

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
        return spsolve(A.T * A + l ** 2 * D.T * D, A.T * g)

    def Tikhonov(self, A, b, l, spacing=None):
        """Solve A * x = b, with Tikhonov regularization parameter l

        Solve the equation diag(f) + l**2 * diag(f).I * D.T * D = g
        where D is the difference operator matrix, and diag the diagonal matrix

        Parameters:
        ----------
        A : np.ndarray, or sparse matrix
        b : np.ndarray
        l : float
            Tikhonov regularization parameter
        Returns
        -------
        np.ndarray
            x
        """

        # Create the matrix A (and its inverse)
        if spacing is None:
            n = A.shape[1]
            D = self.gradient_matrix(n)
        else:
            D = self.gradient_matrix(spacing)

        # Solve the equation
        sol = self._Tikhonov(A, D, l, b)
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
        if hasattr(grid, "__len__"):
            size = len(grid)
        else:
            size = int(grid)

        # https://en.wikipedia.org/wiki/Finite_difference
        # n = self.difference_accuracy
        # factors = [(-1)**i * binom(n, i) for i in range(n + 1)]

        factors = [-1, 2, -1]

        # first order
        # factors = np.array([3, -32, 168, -672, 0, 672, -168, 32, -3]) / 840

        # Why second order factors?
        # factors = np.array([-9, 128, -1008, 8064, -14350, 8064, -1008, 128, -9]) / 5040

        nf = len(factors)
        offsets = np.arange(nf) - nf // 2
        columns = [np.full(size - abs(j), f) for j, f in zip(offsets, factors)]
        columns[1][0] = columns[1][-1] = 1
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

    def best_lambda(self, f, g, wavelength, times):
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
            elif self._method == "Franklin":
                sol = spsolve(A + lamb * D, r)

            x = norm(A * sol - r, 2)
            y = norm(D * sol, 2)
            return x, y

        def func(lamb, ratio, A, D, r, angle=-np.pi / 4):
            """ get "goodness" value for a given lambda using L-parameter """
            progress.update(1)
            x, y = get_point(lamb[0], A, D, r)
            # scale and rotate point
            _, j = rotate(x, y, angle)
            return j

        def rotate(x, y, angle):
            i = x * np.cos(angle) + y * ratio * np.sin(angle)
            j = -x * np.sin(angle) + y * ratio * np.cos(angle)
            return i, j

        # prepare matrices
        w0, A, g = self.prepare_tikhonov(f, g, wavelength, times, reverse=False)
        D = self.gradient_matrix(len(w0))

        # Calculate best lambda
        progress = tqdm(total=500)
        ratio = self.regularization_ratio
        angle = -np.pi / 4

        # Just sample lots of points for a first guess
        ls = np.geomspace(1, 1e6, 300)
        tmp = [get_point(l, A, D, g) for l in tqdm(ls)]
        x = np.array([t[0] for t in tmp])
        y = np.array([t[1] for t in tmp])

        xp, yp = np.copy(x), np.copy(y)

        x, y = rotate(x, y, angle)
        i = np.argmin(y)
        lamb = ls[i]

        # Improve the guess with a minimum solver
        res = minimize(func, x0=[lamb], args=(ratio, A, D, g), method="Nelder-Mead")
        lamb = res.x[0]
        progress.close()

        if self.plot:
            p3 = get_point(lamb, A, D, g)
            plt.plot(xp, yp, "+")
            plt.plot(p3[0], p3[1], "rd")
            # plt.loglog()
            plt.xlabel(r"$||A x - b||_2$")
            plt.ylabel(r"$||D x||_2$")
            plt.show()

        return lamb

    def least_squares(self, f, g, wavelength, wavelength_shifted, regweight):
        wave = wavelength[0]
        p = np.zeros(wavelength.shape)
        mask = np.isfinite(f) & np.isfinite(g)

        def func(x, p):
            for i in range(len(wavelength)):
                p[i] = np.interp(wave, wavelength_shifted[i], x, left=1, right=1)
            return (f * p - g)[mask]

        m = np.count_nonzero(mask)
        n = wavelength.shape[1]
        reg = np.empty(n)
        D = self.gradient_matrix(n)

        def regression(x):
            reg = D * x
            reg = reg ** 2
            return reg

        # TODO: Use sparse jacobian
        res = least_squares(
            func,
            x0=np.ones(n),
            method="trf",
            verbose=2,
            regularization=regression,
            r_scale=regweight,
            args=[p],
            tr_solver="lsmr",
        )

        x0 = res.x
        return x0

    def create_projection_matrix(self, wavelength_reference, wavelength_data):
        w0 = wavelength_reference
        proj = lil_matrix((wavelength_data.size, w0.size))
        digits = np.digitize(wavelength_data, w0)
        for i, (d, w) in tqdm(
            enumerate(zip(digits, wavelength_data)), total=digits.size, leave=False
        ):
            tmp = (w - w0[d - 1]) / (w0[d] - w0[d - 1])
            proj[i, d - 1] = 1 - tmp
            proj[i, d] = tmp
        proj = proj.tocsc()
        return proj

    def shift_wavelength(self, wavelength, times, reverse=False, copy=False, rv=None):
        if copy:
            wavelength = np.copy(wavelength)
        if rv is None:
            rv = self.telescope_frame.to_frame(self.planet_frame, times)
        beta = (rv / c).to_value(1)
        beta *= -1 if reverse else 1
        wavelength *= np.sqrt((1 + beta) / (1 - beta))[:, None]
        return wavelength

    def prepare_tikhonov(self, f, g, wavelength, times, reverse=False, rv=None):
        wavelength = self.shift_wavelength(
            wavelength, times, reverse=reverse, copy=True, rv=rv
        )

        # Make our own wavelength grid
        w0 = []
        diff = np.diff(wavelength[0])
        idx = np.where(diff > 100 * np.median(diff))[0] + 1
        idx = [0, *idx, len(wavelength[0]) - 1]
        for left, right in zip(idx[:-1], idx[1:]):
            w = wavelength[:, left:right]
            w = np.geomspace(w.min(), w.max() + 0.1, num=(right - left) + 1)
            w0 += [w]
        w0 = np.concatenate(w0)

        # Sort by wavelength
        wave = wavelength.ravel()
        f, g = f.ravel(), g.ravel()
        idx = np.argsort(wave)
        wave = wave[idx]
        f, g = f[idx], g[idx]

        # Remove Inf and NaN values
        mask = np.isfinite(f) & np.isfinite(g)
        # mask &= (f != 0) & (g != 0)
        wave = wave[mask]
        f, g = f[mask], g[mask]

        # Setup linear interpolation scheme
        proj = self.create_projection_matrix(w0, wave)
        A = diags(f, 0) * proj
        return w0, A, g

    def solve(
        self,
        times,
        wavelength,
        spectra,
        stellar,
        intensities,
        telluric,
        reverse=False,
        rv=None,
    ):
        """
        Find the least-squares solution to the linear equation
        f * x - g = 0
        """
        wavelength, f, g = self.prepare_fg(
            times, wavelength, spectra, stellar, intensities, telluric
        )

        if self.regularization:
            if self.regularization_weight is None:
                # regweight = 200
                regweight = self.best_lambda(f, g, wavelength, times)
                print("Regularization weight: ", regweight)
            else:
                regweight = self.regularization_weight
        else:
            regweight = 0

        # Each observation will have a different wavelength grid
        # in the planet restframe (since the planet moves quite fast)
        # therefore we use each wavelength point from each observation
        # individually, but we sort them by wavelength
        # so that the gradient, is still only concerned about the immediate
        # neighbours
        # TODO: rv is (approx?) linear in time, so we could try to fit that as well
        # at the cost of having to compute a new wavelength grid everytime
        # although the correlation might not be strong enough to restrain this properly
        if self._method == "lsq":
            wavelength_shifted = self.shift_wavelength(
                wavelength, times, reverse=reverse
            )
            x0 = self.least_squares(f, g, wavelength, wavelength_shifted, regweight)
            wave = wavelength[0]
        elif self._method == "Tikhonov":
            # Run Tikhonov
            w0, A, g = self.prepare_tikhonov(
                f, g, wavelength, times, reverse=reverse, rv=rv
            )
            x0 = self.Tikhonov(A, g, regweight)

        # Normalize x0, each segment individually
        if self.normalize:
            diff = np.diff(wave)
            idx = [0, *np.where(diff > 1000 * np.median(diff))[0], -1]
            for left, right in zip(idx[:-1], idx[1:]):
                x0[left:right] -= np.nanpercentile(x0[left:right], 5)
                x0[left:right] /= np.nanpercentile(x0[left:right], 90)

        spec = Spectrum1D(
            flux=x0 << u.one,
            spectral_axis=w0 << u.AA,
            source="Linear solver",
            description="recovered planet transmission spectrum",
            reference_frame="planet",
            star=self.star,
            planet=self.planet,
            observatory_location=self.detector.observatory,
            datetime=times[0],
            meta={"regularization_weight": regweight},
        )

        return spec
