from .solver import SolverBase
from ..spectrum import Spectrum1D

from scipy.interpolate import UnivariateSpline
from astropy import units as u


class SplineSolver(SolverBase):
    def __init__(
        self, detector, star, planet,
    ):
        super().__init__(detector, star, planet)

    def solve(self, times, wavelength, spectra, stellar, intensities, telluric):
        wave, f, g = self.prepare_fg(
            times, wavelength, spectra, stellar, intensities, telluric
        )
        spl = UnivariateSpline(wave, g / f, s=spectra.shape[1])
        x0 = spl(wave)

        spec = Spectrum1D(
            flux=x0 << u.one,
            spectral_axis=wave << u.AA,
            source="Spline solver",
            description="recovered planet transmission spectrum",
            reference_frame="planet",
            star=self.star,
            planet=self.planet,
            observatory_location=self.detector.observatory,
        )

        return spec
