"""
Option A:
  - resample all spectra to a common wavelength grid
  - add them together
  - fit to that
Pros:
  - easier to normalize
  - radial velocity shift between observations is small (if from the same transit)
Cons:
  - Loose some precision due to resampling

Option B:
  - calculate SME spectra without sampling to wavelength grid
  - do least-squares to each spectrum individually and add them together
Pros:
  - more precise, since no interpolation
Cons:
  - difficult to normalize individual spectra
  - more effort

Lets do Option A!

"""

import numpy as np
from astropy import units as u
from exoorbit.bodies import Star
from pysme.gui import plot_plotly
from pysme.linelist.vald import ValdFile
from pysme.sme import SME_Structure
from pysme.solve import SME_Solver
from pysme.synthesize import Synthesizer, synthesize_spectrum
from tqdm import tqdm

from ..data_modules.sme import SmeStellar, SmeIntensities
from ..data_modules.combine import combine_observations
from ..spectrum import SpectrumArray, SpectrumArrayIO
from .steps import Step, StepIO


class SmeStructureIO(StepIO):
    def save(self, data, filename=None):
        if filename is None:
            filename = self.savefilename
        data.save(filename)

    def load(self, filename=None):
        if filename is None:
            filename = self.savefilename
        data = SME_Structure.load(filename)
        return data


class StellarParametersFirstGuessStep(Step, SmeStructureIO):
    filename = "parameters_first_guess.sme"

    def run(self, spectra, star, detector, linelist):
        blaze = detector.blaze

        spectrum = combine_observations(spectra)
        spectrum = spectrum[0]
        sme = self.create_first_guess(
            spectrum, star, blaze, linelist, detector=detector
        )

        sme.cscale_flag = "fix"
        sme.vrad_flag = "fix"
        self.save(sme)
        return sme

    def create_first_guess(
        self, spectrum, star, blaze, linelist, detector, uncs=None,
    ):
        print("Extracting stellar parameters...")

        # Create SME structure
        print("Preparing PySME structure")
        sme = SME_Structure()
        sme.wave = [wave.to_value(u.AA) for wave in spectrum.wavelength]
        sme.spec = [spec.to_value(1) for spec in spectrum.flux]
        if spectrum.uncertainty is not None:
            sme.uncs = [unc.array * unc.unit.to(1) for unc in spectrum.uncertainty]

        sme.teff = star.teff.to_value(u.K)
        sme.logg = star.logg.to_value(1)
        sme.monh = star.monh.to_value(1)
        sme.vturb = star.vturb.to_value(u.km / u.s)

        sme.abund = "solar"
        sme.linelist = ValdFile(linelist)

        sme.atmo.source = "marcs"
        sme.atmo.method = "grid"

        nlte = None
        if nlte is not None:
            for elem, grid in nlte.items():
                sme.nlte.set_nlte(elem, grid)

        sme.cscale_flag = "none"
        sme.normalize_by_continuum = True
        sme.vrad_flag = "fix"
        sme.vrad = star.radial_velocity.to_value("km/s")

        if detector is not None:
            sme.iptype = "gauss"
            sme.ipres = detector.resolution

        # Create an initial spectrum using the nominal values
        # This also determines the radial velocity
        print("Determine the radial velocity using the nominal stellar parameters")
        synthesizer = Synthesizer()
        sme = synthesizer.synthesize_spectrum(sme)
        return sme


class StellarParametersFitStep(Step, StepIO):
    filename = "star.yaml"

    def run(self, stellar_parameters_first_guess, star):
        sme = stellar_parameters_first_guess["sme"]
        sme, star = self.fit_observation(sme, star)
        self.save(star)
        return star

    def fit_observation(
        self,
        sme,
        star,
        segments="all",
        parameters=["teff", "logg", "monh", "vsini", "vmac", "vmic"],
    ):
        # Fit the observation with SME
        print("Fit stellar spectrum with PySME")
        # sme.cscale_flag = "linear"
        # sme.cscale_type = "mask"
        # sme.vrad_flag = "whole"

        solver = SME_Solver()
        sme = solver.solve(sme, param_names=parameters, segments=segments)

        fig = plot_plotly.FinalPlot(sme)
        fig.save(filename="solved.html")

        # Save output
        print("Save results")
        for param in parameters:
            unit = getattr(star, param).unit
            print(f"{param}: {sme[param]} {unit}")
            setattr(star, param, sme[param] * unit)

        # TODO: + barycentric correction
        star.radial_velocity = sme.vrad[0] << (u.km / u.s)

        return sme, star

    def save(self, data, filename=None):
        if filename is None:
            filename = self.savefilename
        star.save(filename)

    def load(self, filename=None):
        if filename is None:
            filename = self.savefilename
        star = Star.load(filename)
        return star


class StellarStep(Step, SpectrumArrayIO):
    filename = "stellar_general.npz"
    source = None

    def run(self, star, detector, linelist, spectra):
        times = spectra.datetime
        wrange = detector.regions
        stellar = self.create_stellar(
            wrange, spectra, times, method=self.source, star=star, linelist=linelist
        )
        self.save(stellar)
        return stellar

    def create_stellar(self, wrange, spectra, times, method, **kwargs):
        print("Creating stellar...")
        if method == "sme":
            stellar = SmeStellar(**kwargs, normalize=True)
        elif method == "combine":
            stellar = CombineStellar(spectra, **kwargs)
            return stellar.combined, stellar.telluric
        else:
            raise ValueError

        reference_frame = spectra.reference_frame
        result = []
        for i, time in tqdm(enumerate(times), total=len(times)):
            wave = [
                spectra.wavelength[i][low:top]
                for low, top in zip(spectra.segments[:-1], spectra.segments[1:])
            ]
            # wave = spectra[i].wavelength
            spec = stellar.get(wrange, time)
            spec = spec.shift(reference_frame, inplace=True)
            spec = spec.resample(wave, method="linear")
            result += [spec]

        result = SpectrumArray(result)
        return result


class StellarSpectrumStep(StellarStep, SpectrumArrayIO):
    filename = "stellar.npz"
    source = "sme"

class StellarSpectrumCombinedStep(StellarStep):
    filename_stellar = "stellar_spectrum_combined.npz"
    filename_telluric = "telluric_combined.npz"
    source = "combined"

    def run(self, detector, normalized_observation, telluric, stellar_spectrum):
        wrange = detector.regions
        times = normalized_observation.datetime

        stellar_combined, telluric_combined = create_stellar(
            wrange,
            normalized_observation,
            times,
            method="combine",
            telluric=telluric,
            detector=detector,
            stellar=stellar_spectrum,
        )
        self.save(stellar_combined, self.filename_stellar)
        self.save(tellurics_combined, self.filename_tellurics)
        return stellar_combined, telluric_combined

    def load(self, filename_stellar=None, filename_telluric=None):
        if filename_stellar is None:
            filename_stellar = self.filename_stellar
        if filename_telluric is None:
            filename_telluric = self.filename_telluric

        stellar = super().load(filename_stellar)
        telluric = super().load(filename_telluric)
        return stellar, telluric
