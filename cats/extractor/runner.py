# Standard library imports
import inspect
import logging
import os
from os.path import dirname, exists, join

# 3rd party imports
import numpy as np
from astropy import units as u
from exoorbit import Orbit
from exoorbit.bodies import Planet, Star

# CATS modules
from ..data_modules.stellar_db import StellarDb
from .step_collect import CollectObservationsStep
from .step_cross_correlation import (
    CrossCorrelationReferenceStep,
    CrossCorrelationStep,
    PlanetAtmosphereReferencePetitRadtransStep,
    PlanetRadialVelocityStep,
)
from .step_intensities import SpecificIntensitiesSmeStep
from .step_normalize import NormalizeObservationsStep
from .step_planet import PlanetParametersStep
from .step_solve import PlanetAbsorptionAreaStep, SolveProblemStep
from .step_stellar import (
    StellarParametersFirstGuessStep,
    StellarParametersFitStep,
    StellarSpectrumCombinedStep,
    StellarSpectrumStep,
)
from .step_telluric import TelluricAirmassStep, TelluricSpaceStep, TelluricTapasStep
from .steps import Step, StepIO

logger = logging.getLogger(__name__)


class CatsRunner:
    names_of_steps = {
        "spectra": CollectObservationsStep,
        "telluric": TelluricAirmassStep,
        "telluric_tapas": TelluricTapasStep,
        "telluric_space": TelluricSpaceStep,
        "stellar_parameters_first_guess": StellarParametersFirstGuessStep,
        "star": StellarParametersFitStep,
        "stellar_spectrum": StellarSpectrumStep,
        "normalized_observation": NormalizeObservationsStep,
        "stellar_spectrum_combined": StellarSpectrumCombinedStep,
        "planet": PlanetParametersStep,
        "specific_intensities": SpecificIntensitiesSmeStep,
        "planet_reference_spectrum": PlanetAtmosphereReferencePetitRadtransStep,
        "cross_correlation_reference": CrossCorrelationReferenceStep,
        "cross_correlation": CrossCorrelationStep,
        "planet_radial_velocity": PlanetRadialVelocityStep,
        "planet_area": PlanetAbsorptionAreaStep,
        "solve_problem": SolveProblemStep,
    }
    step_order = {
        "spectra": 10,
        "telluric_airmass": 20,
        "telluric_tapas": 30,
        "telluric_space": 40,
        "stellar_parameters_first_guess": 50,
        "star": 60,
        "stellar_spectrum": 70,
        "normalized_observation": 80,
        "stellar_spectrum_combined": 90,
        "planet": 100,
        "specific_intensities": 110,
        "planet_reference_spectrum": 120,
        "cross_correlation_reference": 130,
        "cross_correlation": 140,
        "planet_radial_velocity": 150,
        "planet_area": 160,
        "solve_problem": 500,
    }

    def __init__(
        self,
        detector,
        star,
        planet,
        linelist,
        base_dir=None,
        raw_dir=None,
        medium_dir=None,
        done_dir=None,
    ):
        #:Detector: The detector/instrument performing the measurements
        self.detector = detector
        #:EarthLocation: The observatory coordinates
        self.observatory = self.detector.observatory

        if not isinstance(star, Star):
            sdb = StellarDb()
            star = sdb.get(star)
        #:Star: The parameters of the star
        self.star = star

        if not isinstance(planet, Planet):
            planet = self.star.planets[planet]
        #:Planet: The parameters of the planet
        self.planet = planet

        self.linelist = linelist
        self.orbit = Orbit(self.star, self.planet)

        if base_dir is None:
            base_dir = os.getcwd()
        if raw_dir is None:
            self.raw_dir = join(base_dir, "raw")
        else:
            self.raw_dir = raw_dir

        if medium_dir is None:
            self.medium_dir = join(base_dir, "medium")
        else:
            self.medium_dir = medium_dir

        if done_dir is None:
            self.done_dir = join(base_dir, "done")
        else:
            self.done_dir = done_dir

        self.data = {
            "raw_dir": self.raw_dir,
            "medium_dir": self.medium_dir,
            "done_dir": self.done_dir,
            "star": self.star,
            "planet": self.planet,
            "detector": self.detector,
            "observatory": self.observatory,
            "linelist": self.linelist,
        }

    def run(self, steps):
        # Make sure the directories exists
        for d in [self.medium_dir, self.done_dir]:
            os.makedirs(d, exist_ok=True)

        if steps == "all":
            steps = list(self.step_order.keys())
        steps = list(steps)
        # Order steps in the best order
        steps = sorted(steps, key=lambda s: self.step_order[s])

        # Run individual steps
        for step in steps:
            self.run_module(step)

        return self.data

    def run_module(self, step, load=False):
        # The Module this step is based on (An object of the Step class)
        module = self.names_of_steps[step](self.raw_dir, self.medium_dir, self.done_dir)

        # Load the dependencies necessary for loading/running this step
        # We determine this through introspection
        members = inspect.getmembers(module.__class__.run)
        members = [m for m in members if m[0] == "__code__"][0][1]
        # We skip the first element, as that is 'self'
        dependencies = inspect.getargs(members).args[1:]
        # Then we get all the data from other steps if necessary
        for dependency in dependencies:
            if dependency not in self.data.keys():
                self.data[dependency] = self.run_module(dependency, load=True)
        args = {d: self.data[d] for d in dependencies}

        # Try to load the data, if the step is not specifically given as necessary
        # If the intermediate data is not available, run it normally instead
        # But give a warning
        if load:
            try:
                logger.info("Loading data from step '%s'", step)
                data = module.load()
            except FileNotFoundError:
                logger.warning(
                    "Intermediate File(s) for loading step %s not found. Running it instead.",
                    step,
                )
                data = self.run_module(step, load=False)
        else:
            logger.info("Running step '%s'", step)
            data = module.run(**args)

        self.data[step] = data
        return data
