import numpy as np
import astropy.units as u

from .datasource import DataSource, StellarIntensities
from ..spectrum import Spectrum1D

from pysme.sme import SME_Structure
from pysme.synthesize import Synthesizer


class SmeStellar(DataSource):
    def __init__(self, star):
        super().__init__()
        self.star = star
        self.wave = wave

    def get(self, wave):
        sme = SME_Structure()
        sme.teff = self.star.teff
        sme.logg = self.star.logg
        sme.monh = self.star.monh
        # TODO other stellar parameters
        
        sme.wave = wave

        sme.cscale_flag = "none"
        sme.normalize_by_continuum = False
        sme.vrad_flag = "none"

        synthesizer = Synthesizer()
        sme = synthesizer.synthesize_spectrum(sme)

        wave = sme.wave[0] << u.AA
        spec = sme.synth[0] << u.Unit(1)

        synth = Spectrum1D(
            flux=spec,
            spectral_axis=wave,
            reference_frame="barycentric",
            star=self.star,
            source="sme",
            description="synthetic stellar spectrum",
            citation=sme.citation(format="bibtex")
        )
        return synth


class SmeIntensities(StellarIntensities):
    def get(self, wave, mu, mode="core"):
        sme = SME_Structure()
        sme.teff = self.star.teff
        sme.logg = self.star.logg
        sme.monh = self.star.monh
        
        sme.wave = wave
        sme.mu = mu

        sme.cscale_flag = "none"
        sme.normalize_by_continuum = False
        sme.vrad_flag = "none"

        sme.specific_intensities_only = True

        synthesizer = Synthesizer()
        wave, spec, cont = synthesizer.synthesize_spectrum(sme)

        wave = wave << u.AA
        spec = spec << u.Unit(1)

        synth = Spectrum1D(
            flux=spec,
            spectral_axis=wave,
            reference_frame="barycentric",
            star=self.star,
            planet=self.planet,
            source="sme",
            description="stellar specific intensities",
            citation=sme.citation(format="bibtex")
        )

        # TODO mode differences?
        # Integrate over a range of mu values for each given mu value?

        return synth
