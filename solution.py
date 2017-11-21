"""
Solve the linearized minimization Problem Phi = sum(G*P - F) + lam * R
"""

import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import minimize

class solution:
    """ Wrapper class for the functions """

    def __init__(self, dtype=np.float):
        self.dtype = dtype

    # TODO try to find best value for lambda
    # Brute Force Solution: try different values for lambda and find the best
    # What is the best lambda ??? Which metric is used to determine that?

    def solve(self, wl, f, g, lamb):
        """
        Solve the mimimazation problem to find the planetary spectrum
        F*x - G = 0
        wl_grid: Wavelength scale
        F: intermediary product F
        G: intermediary product G
        lam: regularization parameter lambda
        """
        if isinstance(lamb, np.ndarray) and len(lamb) == 1:
            lamb = lamb[0]

        if isinstance(lamb, (int, float)):
            lamb = np.full(len(wl), lamb, dtype=self.dtype)
        a, c = np.zeros(len(wl), dtype=np.float32), np.zeros(len(wl), dtype=self.dtype)
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

    def solve2(self, wl, f, g, l):
        """ Using Sparse Matrixes, experimental might be faster? """
        from scipy.sparse import diags, csc_matrix
        from scipy.sparse.linalg import spsolve
        n = len(wl)
        a = c = np.full(n-1, -l)
        b = f[0] + 2 * l
        b[0] -= l
        b[-1] -= l
        A = csc_matrix(diags([a, b, c], offsets=[-1, 0, 1]))
        return spsolve(A, g[0])

"""
#This just fits the input spectrum, independant of G and F
print('Alternative questionable apporach')
exo2 = pd.read_table(os.path.join(
    input_dir, par['file_atmosphere']), header=None, delim_whitespace=True).values
exo2 = interp1d(exo2[:, 0], exo2[:, 1], kind=config['interpolation_method'],
                fill_value='extrapolate')(wl_grid)
exo2 = exo2 * par['h_atm'] - par['h_atm']

# makes no difference
#G = np.random.rand(*G.shape)
Fexo = exo2 * G
lambdaexo = 1500 * 3 / par['n_exposures']
exo = solve(wl_grid, Fexo, G, lambdaexo)

# Normalize
exo = (exo - np.min(exo)) / np.max(exo - np.min(exo))
exo2 = (exo2 - np.min(exo2)) / np.max(exo2 - np.min(exo2))

plt.plot(wl_grid, exo2)
plt.plot(wl_grid, exo, 'r')
plt.show()
"""