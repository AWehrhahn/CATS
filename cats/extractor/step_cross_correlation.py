import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from tqdm import tqdm

from flex.flex import FlexFile

from scipy.ndimage.filters import gaussian_filter1d

from astropy import units as u
from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc

from ..pysysrem.sysrem import sysrem
from ..spectrum import Spectrum1D, Spectrum1DIO, SpectrumArray, SpectrumArrayIO
from .steps import Step, StepIO


class PlanetAtmosphereReferencePetitRadtransStep(Step, Spectrum1DIO):
    filename = "reference_petitRADTRANS.npz"

    def radtrans(self, wrange, star, planet):
        # 0.8 to 5
        wmin = wrange[0].to_value("um")
        wmax = wrange[1].to_value("um")
        # Initialize atmosphere
        # including the elements in the atmosphere
        atmosphere = Radtrans(
            line_species=["H2O"],
            # line_species=["H2O", "CO_all_iso", "CH4", "CO2", "Na", "K"],
            rayleigh_species=["H2", "He"],
            continuum_opacities=["H2-H2", "H2-He"],
            wlen_bords_micron=[wmin, wmax],
            mode="lbl",
        )

        # Define planet parameters
        # Planet radius
        R_pl = planet.radius.to_value("cm")
        # R_pl = 1.838 * nc.r_jup_mean
        # surface gravity
        # gravity = 1e1 ** 2.45
        gravity = planet.surface_gravity.to_value("cm/s**2")
        # reference pressure (for the surface gravity and radius)
        # TODO: ????
        P0 = 0.01

        # Pressure in bar
        # has to be equispaced in log
        print("Setup atmosphere pressures")
        pressures = np.logspace(-6, 2, 100)
        atmosphere.setup_opa_structure(pressures)

        # Define temperature pressure profile
        kappa_IR = 0.01  # opacity in the IR
        gamma = 0.4  # ratio between the opacity in the optical and the IR
        T_int = 200.0  # Internal temperature
        # T_equ = 1500.0
        T_equ = planet.teff_from_stellar(star.teff).to_value("K")
        temperature = nc.guillot_global(
            pressures, kappa_IR, gamma, gravity, T_int, T_equ
        )

        # Define mass fractions
        mass_fractions = {}
        mass_fractions["H2"] = 0.74 * np.ones_like(temperature)
        mass_fractions["He"] = 0.24 * np.ones_like(temperature)
        mass_fractions["H2O"] = 0.001 * np.ones_like(temperature)
        # mass_fractions["CH4"] = 0.000001 * np.ones_like(temperature)
        # mass_fractions["CO_all_iso"] = 0.01 * np.ones_like(temperature)
        # mass_fractions["CO2"] = 0.00001 * np.ones_like(temperature)
        # mass_fractions["Na"] = 0.00001 * np.ones_like(temperature)
        # mass_fractions["K"] = 0.000001 * np.ones_like(temperature)

        MMW = 2.33 * np.ones_like(temperature)

        # Calculate transmission spectrum
        print("Calculate transmission Spectrum")
        atmosphere.calc_transm(
            temperature, mass_fractions, gravity, MMW, R_pl=R_pl, P0_bar=P0
        )
        # atmosphere.calc_flux(temperature, mass_fractions, gravity, MMW)
        wave = nc.c / atmosphere.freq / 1e-4
        flux = 1 - (atmosphere.transm_rad / nc.r_sun) ** 2

        wave = wave << u.um
        flux = flux << u.one

        spec = Spectrum1D(
            spectral_axis=wave,
            flux=flux,
            description="petitRADTRANS transmission spectrum",
            source="petitRADTRANS",
            reference_frame="barycentric",
            star=star,
            planet=planet,
        )

        return spec

    def run(self, star, planet, detector):
        rv_padding = self.rv_padding
        wrange = detector.regions
        wmin = min([wr.lower for wr in wrange])
        wmax = max([wr.upper for wr in wrange])
        # Apply possible radial velocity tolerance of 200 km/s
        wmin *= 1 - rv_padding / 3e5
        wmax *= 1 + rv_padding / 3e5
        ref = self.radtrans([wmin, wmax], star, planet)
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
        rv_range = self.rv_range
        rv_points = self.rv_points

        idx_middle = len(spectra) // 2

        ref = planet_reference_spectrum
        ref.star = star
        ref.observatory_location = observatory
        ref.datetime = spectra.datetime[idx_middle]
        ref_wave = np.copy(spectra.wavelength[idx_middle])
        reference = np.zeros((rv_points, ref_wave.size))

        rv = np.linspace(-rv_range, rv_range, num=rv_points)
        rv = rv << (u.km / u.s)

        for i in tqdm(range(rv_points)):
            tmp = ref.shift("barycentric", rv=rv[i], inplace=False)
            tmp = tmp.resample(ref_wave, inplace=False, method="linear")
            reference[i] = np.nan_to_num(tmp.flux.to_value(1))

        reference = reference << u.one
        reference = Spectrum1D(
            spectral_axis=ref_wave, flux=reference, datetime=spectra.datetime[len(spectra) // 2]
        )
        # We are only looking at the difference between the median and the observation
        # Thus additional absorption would result in a negative signal at points of large absorption
        reference.flux[:] -= 1

        self.save(reference)
        return reference


class CrossCorrelationStep(Step, StepIO):
    filename = "cross_correlation.npz"

    def run(self, spectra, cross_correlation_reference):
        max_nsysrem = self.max_sysrem_iterations
        max_nsysrem_after = self.max_sysrem_iterations_afterwards
        rv_range = self.rv_range
        rv_points = self.rv_points

        reference = cross_correlation_reference

        # entries past 90 are 'weird'
        flux = spectra.flux.to_value(1)
        flux = flux
        unc = spectra.uncertainty.array

        correlation = {}
        for n in tqdm(range(max_nsysrem), desc="Sysrem N"):
            corrected_flux = sysrem(flux, num_errors=n, errors=unc)

            # Mask strong tellurics
            std = np.nanstd(corrected_flux, axis=0)
            std[std == 0] = 1
            corrected_flux /= std

            # Observations 90 to 101 have weird stuff
            corr = np.zeros((len(spectra), rv_points))
            for i in tqdm(range(len(spectra)), leave=False, desc="Observation"):
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
            for i in tqdm(
                range(max_nsysrem_after),
                leave=False,
                desc="Sysrem on Cross Correlation",
            ):
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

    def plot(self, data, sysrem_iterations=2, sysrem_iterations_afterwards=4):
        if sysrem_iterations_afterwards is not None:
            corr = data[f"{sysrem_iterations}.{sysrem_iterations_afterwards}"]
        else:
            corr = data[f"{sysrem_iterations}"]

        vmin, vmax = np.nanpercentile(corr, (1, 99))
        plt.imshow(corr, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)

        plt.title(f"Cross-Correlation for SYSREM Iteration: {sysrem_iterations}")

        xticks = plt.xticks()[0][1:-1]
        xticks_labels = xticks - self.rv_range
        plt.xticks(xticks, labels=xticks_labels)

        plt.xlabel("rv [km/s]")
        plt.ylabel("#Obs")
        plt.show()


class PlanetRadialVelocityStep(Step, StepIO):
    filename = "planet_radial_velocity.flex"

    def run(self, cross_correlation, spectra):
        rv_range = self.rv_range
        rv_points = self.rv_points
        n = self.sysrem_iterations
        i = self.sysrem_iterations_afterwards

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
        lower, upper = int(0.2 * n_obs), int(0.8 * n_obs)
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
