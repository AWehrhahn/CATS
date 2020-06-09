"""
Based on: https://onlinelibrary-wiley-com.ezproxy.its.uu.se/doi/full/10.1002/wics.1427

4 steps to solve bayesian inversion problem:
  - Constructing an informative and computationally feasible prior;
  - Constructing the likelihood using the forward model and information about the noise;
  - Forming the posterior density using Bayes’ formula;
  - Extracting information from the posterior.

"""
import numpy as np
import emcee
from scipy.sparse import diags

from .solver import SolverBase
from ..spectrum import Spectrum1D


class BayesSolver(SolverBase):
    def __init__(self, detector, star, planet):
        super().__init__(detector, star, planet)

        self.difference_accuracy = 8
        self.regweight = 0

        self.grad = None
        self.gamma = 0.5

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

    def smoothness_prior(self, theta):
        # Eq 8
        # grad: gradient matrix
        # x: planet spectrum
        # y: wavelength
        p = -1 / (2 * self.gamma ** 2) * np.sum((self.grad * theta) ** 2)
        return p

    def mattern_whittle_prior(self):
        # np.pow(-D + lamb**2, beta) ~ y * W
        pass

    def log_posterior(self, theta, wave, f, g):
        model = f * theta
        prior = self.smoothness_prior(theta)
        if not np.isfinite(prior):
            return -np.inf
        n = len(model)
        chisq = np.sum((model - g) ** 2)
        reg = np.sum((self.grad * model) ** 2)
        prob = -0.5 * (chisq + 1 / self.regweight * reg) - n / 2 * np.log(
            self.regweight
        )
        return prior + prob

    def solve(
        self, times, wavelength, spectra, stellar, intensities, telluric, regweight=1
    ):

        print(
            "Bayesian solver is experimental and will probably not work for large problems"
        )

        wave, f, g = self.prepare_fg(
            times, wavelength, spectra, stellar, intensities, telluric
        )

        self.regweight = regweight
        self.grad = self.gradient_matrix(wave)

        # TODO: MCMC for large dimensions? Not great, needs way too much memory (among other problems)
        # What to do instead? Gaussian Processes???? How do they even work?
        nwalkers = 32
        npoints = wave.size

        pos = np.random.random((nwalkers, npoints))
        # pos = pos + 1e-4 * np.random.randn(nwalkers, npoints)

        sampler = emcee.EnsembleSampler(
            nwalkers, npoints, self.log_posterior, args=(wave, f, g)
        )
        sampler.run_mcmc(pos, 5000, progress=True, skip_initial_state_check=True)

        flat_samples = sampler.get_chain(discard=100, flat=True)
        print(flat_samples.shape)

        x0 = np.median(flat_samples, axis=0)

        spec = Spectrum1D(
            flux=x0 << u.one,
            spectral_axis=wave << u.AA,
            source="Bayes solver",
            description="recovered planet transmission spectrum",
            reference_frame="planet",
            star=self.star,
            planet=self.planet,
            observatory_location=self.detector.observatory,
        )

        return spec
