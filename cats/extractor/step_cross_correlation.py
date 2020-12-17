import numpy as np
import radtrans as petit_radtrans
from scipy.optimize import least_squares

from flex.flex import FlexFile

from ..pysysrem.sysrem import sysrem
from ..spectrum import Spectrum1D, Spectrum1DIO, SpectrumArray, SpectrumArrayIO
from .steps import Step


class PlanetAtmosphereReferencePetitRadtransStep(Step, Spectrum1DIO):
    filename = "reference_petitRADTRANS.npz"

    def run(self, star, planet, detector):
        wrange = detector.regions
        wmin = min([wr.lower for wr in wrange])
        wmax = max([wr.upper for wr in wrange])
        # Apply possible radial velocity tolerance of 200 km/s
        wmin *= 1 - 200 / 3e5
        wmax *= 1 + 200 / 3e5
        ref = petit_radtrans.radtrans([wmin, wmax], star, planet)
        ref.star = star
        ref.planet = planet

        # Rescaled ref to 0 to 1
        f = np.sqrt(1 - ref.flux)
        f -= f.min()
        f /= f.max()
        f = 1 - f ** 2
        ref.flux[:] = f

        self.save(ref)
        return ref


class CrossCorrelationReferenceStep(Step, Spectrum1DIO):
    filename = "reference_petitRADTRANS.fits"

    def run(self, planet_reference_spectrum, star, observatory, spectra):
        rv_range = 100
        rv_points = 201

        ref = planet_reference_spectrum
        ref.star = star
        ref.observatory_location = observatory
        ref.datetime = spectra.datetime[50]
        ref_wave = np.copy(spectra.wavelength[50])
        reference = np.zeros((rv_points, ref_wave.size))

        rv = np.linspace(-rv_range, rv_range, num=rv_points)
        rv = rv << (u.km / u.s)

        for i in tqdm(range(rv_points)):
            tmp = ref.shift("barycentric", rv=rv[i], inplace=False)
            tmp = tmp.resample(ref_wave, inplace=False, method="linear")
            reference[i] = np.nan_to_num(tmp.flux.to_value(1))

        reference = reference << u.one
        reference = Spectrum1D(
            spectral_axis=ref_wave, flux=reference, datetime=spectra.datetime[50]
        )
        # We are only looking at the difference between the median and the observation
        # Thus additional absorption would result in a negative signal at points of large absorption
        reference.flux[:] -= 1

        self.save(reference)
        return reference


class CrossCorrelationStep(Step, StepIO):
    filename = "cross_correlation.npz"

    def run(self, spectra, cross_correlation_reference):
        max_nsysrem = 10
        rv_range = 100
        rv_points = 201

        reference = cross_correlation_reference

        # entries past 90 are 'weird'
        flux = spectra.flux.to_value(1)
        flux = flux[:90]
        unc = spectra.uncertainty.array[:90]

        correlation = {}
        for n in tqdm(range(max_nsysrem), desc="Sysrem N"):
            corrected_flux = sysrem(flux, num_errors=n, errors=unc)

            # Mask strong tellurics
            std = np.nanstd(corrected_flux, axis=0)
            std[std == 0] = 1
            corrected_flux /= std

            # Observations 90 to 101 have weird stuff
            corr = np.zeros((90, rv_points))
            for i in tqdm(range(90), leave=False, desc="Observation"):
                for j in tqdm(range(rv_points), leave=False, desc="radial velocity",):
                    for left, right in zip(spectra.segments[:-1], spectra.segments[1:]):
                        m = np.isnan(corrected_flux[i, left:right])
                        m |= np.isnan(reference.flux[j, left:right].to_value(1))
                        m = ~m
                        # Cross correlate!
                        corr[i, j] += np.correlate(
                            corrected_flux[i, left:right][m],
                            reference.flux[j, left:right][m].to_value(1),
                            "valid",
                        )
                        # Normalize to the number of data points used
                        corr[i, j] *= m.size / np.count_nonzero(m)

            correlation[f"{n}"] = np.copy(corr)
            for i in tqdm(range(10), leave=False, desc="Sysrem on Cross Correlation"):
                correlation[f"{n}.{i}"] = sysrem(np.copy(corr), i)

        self.save(correlation)
        return correlation

    def save(self, data, filename=None):
        if filename is None:
            filename = self.savefilename
        np.savez(filename, **data)

    def load(self, filename=None):
        if filename is None:
            filename = self.savefilename
        data = np.load(filename)
        return data


class PlanetRadialVelocityStep(Step, StepIO):
    filename = "planet_radial_velocity.flex"

    def run(self, cross_correlation, spectra):
        rv_range = 100
        rv_points = 201
        n, i = 2, 4

        corr = cross_correlation[f"{n}.{i}"]
        # Remove large scale variation
        tmp = np.copy(corr)
        tmp -= gaussian_filter1d(corr, 20, axis=0)
        tmp -= gaussian_filter1d(corr, 20, axis=1)
        tmp -= np.nanmedian(tmp)
        corr = tmp

        # Fit the detected cross correlation signal with a model
        # TODO: find decent initial values on your own
        # TODO: maybe use MCMC?
        n_obs = spectra.shape[0]
        A = np.nanpercentile(corr, 99)
        # This starting value is very important!!!
        # v_sys = star.radial_velocity.to_value("km/s")
        v_sys = -35  # Star radial velocity + barycentric correction
        v_planet = 30 / 60
        sig = 2
        lower, upper = 20, 80
        x0 = [v_sys, v_planet, sig, A]
        x = np.linspace(-rv_range, rv_range + 1, rv_points)

        def gaussian(x, A, mu, sig):
            return A * np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))

        def model_func(x0):
            mu, shear, sig, A = x0
            model = np.zeros_like(corr)
            for i in range(lower, upper):
                mu_prime = mu + shear * (i - n_obs // 2)
                model[i] = gaussian(x, A, mu_prime, sig)
            return model

        def fitfunc(x0):
            model = model_func(x0)
            resid = model - corr
            return resid.ravel()

        res = least_squares(
            fitfunc,
            x0=x0,
            loss="soft_l1",
            bounds=[[-rv_range, 0, 1, 1], [rv_range, 2, 5, 200]],
            x_scale="jac",
            ftol=None,
        )
        model = model_func(res.x)
        v_sys = res.x[0] << (u.km / u.s)
        v_planet = res.x[1] * (np.arange(n_obs) - n_obs // 2)
        # v_planet = -(v_sys + (v_planet << (u.km / u.s)))
        v_planet = -v_planet << (u.km / u.s)

        data = {"rv_system": v_sys, "rv_planet": v_planet}

        self.save(data)
        return data

    def save(self, data, filename=None):
        if filename is None:
            filename = self.savefilename
        ff = FlexFile(header=data)
        ff.write(filename)

    def load(self, filename=None):
        if filename is None:
            filename = self.savefilename
        ff = FlexFile.read(filename)
        return ff.header
