import numpy as np
from tqdm import tqdm

from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

from .steps import Step
from ..spectrum import SpectrumArrayIO


class NormalizeObservationsStep(Step, SpectrumArrayIO):
    filename = "normalized.flex"

    def normalize_observation(
        self,
        spectra,
        stellar,
        telluric,
        detector,
        stellar_broadening,
        telluric_broadening,
    ):
        sold = deepcopy(spectra)
        spectra = deepcopy(spectra)
        sort = np.argsort(spectra.datetime)

        for j in tqdm(range(len(spectra)), leave=False, desc="Spectrum"):
            wave = spectra.wavelength[j].to_value(u.AA)
            norm = spectra.flux[j].to_value(u.one)
            cont = np.ones_like(norm)
            for left, right in tqdm(
                zip(spectra.segments[:-1], spectra.segments[1:]),
                leave=False,
                desc="Segment",
                total=spectra.nseg,
            ):

                sflux = gaussian_filter1d(
                    stellar.flux[j, left:right].to_value(u.one), stellar_broadening
                )
                tflux = gaussian_filter1d(
                    telluric.flux[j, left:right].to_value(u.one), telluric_broadening
                )
                simulation = sflux * tflux

                x = wave[left:right]
                y = norm[left:right]
                yp = simulation

                mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(yp)
                x, y, yp = x[mask], y[mask], yp[mask]
                if len(x) == 0:
                    continue
                x0 = x[0]
                x -= x0

                def func(x, *c):
                    return y * np.polyval(c, x)

                deg = 5
                p0 = np.zeros(deg + 1)
                p0[-1] = 1 / np.mean(y)
                popt, pcov = curve_fit(func, x, yp, p0=p0)

                # For debugging
                # plt.plot(x, y * np.polyval(popt, x), label="observation")
                # plt.plot(x, yp, label="model")
                # plt.show()

                xs = wave[left:right] - x0
                value = np.polyval(popt, xs)
                spectra.flux[j, left:right] *= value
                if spectra.uncertainty is not None:
                    spectra.uncertainty.array[j, left:right] *= value

        return spectra

    def normalize(self, spectra, stellar, telluric, detector):
        # Also broadening is matched to the observation
        # telluric and stellar have independant broadening factors
        sflux = stellar.flux
        tflux = telluric.flux
        stellar_broadening = 1
        telluric_broadening = 1

        for _ in tqdm(range(3), leave=False, desc="Iteration"):
            normalized = self.normalize_observation(
                spectra,
                stellar,
                telluric,
                detector,
                stellar_broadening,
                telluric_broadening,
            )

            tmp = sflux * tflux
            mask = np.isfinite(tmp)
            func = (
                lambda s: gaussian_filter1d(telluric.flux[mask], abs(s[1]))
                * gaussian_filter1d(stellar.flux[mask], abs(s[0]))
                - normalized.flux[mask]
            )
            res = least_squares(func, x0=[1, 1])
            stellar_broadening = abs(res.x[0].to_value(1))
            telluric_broadening = abs(res.x[1].to_value(1))

            detector.spectral_broadening = stellar_broadening
            tflux = gaussian_filter1d(telluric.flux, telluric_broadening) << u.one
            sflux = gaussian_filter1d(stellar.flux, stellar_broadening) << u.one

        return normalized, stellar_broadening, telluric_broadening

    def run(self, spectra, stellar_spectrum, telluric, detector):
        normalized, stellar_broadening, telluric_broadening = self.normalize(
            spectra, stellar_spectrum, telluric, detector
        )
        normalized.meta["stellar_broadening"] = stellar_broadening
        normalized.meta["telluric_broadening"] = telluric_broadening
        self.save(normalized)
        broadening = {"stellar": stellar_broadening, "telluric": telluric_broadening}
        return normalized, broadening

    def load(self, filename=None):
        normalized = super().load(filename)
        broadening = {
            "stellar": normalized.meta["stellar_broadening"],
            "telluric": normalized.meta["telluric_broadening"],
        }
        return normalized, broadening
