import numpy as np
import astropy.units as u

from .datasource import DataSource, StellarIntensities
from ..spectrum import Spectrum1D

from pysme.sme import SME_Structure
from pysme.synthesize import Synthesizer
from pysme.linelist.vald import ValdFile


def round_to_nearest(value, options):
    value = np.atleast_2d(value).T
    options = np.asarray(options)

    diff = np.abs(value - options)
    sort = np.argsort(diff)
    nearest = options[sort[:, 0]]
    if value.size == 1:
        return nearest[0]
    return nearest

class SmeStellar(DataSource):
    def __init__(self, star, abundance="solar", linelist=None, atmosphere="marcs", nlte=None):
        super().__init__()
        self.star = star
        self.abund = abundance
        self.linelist = linelist
        self.atmosphere = atmosphere
        self.nlte = nlte

        if self.atmosphere == "marcs":
            vturb = self.star.vturb.to_value(u.km / u.s)
            # round to nearest option
            vturb = round_to_nearest(vturb, [0, 1, 2, 3, 4, 5])
            self.atmosphere = f"marcs2012p_t{vturb:1.1f}.sav"

    def get(self, wave):
        sme = SME_Structure()
        sme.teff = self.star.teff.to_value(u.K)
        sme.logg = self.star.logg.to_value(u.one)
        sme.monh = self.star.monh.to_value(u.dex)
        # TODO other stellar parameters
        
        sme.vturb = self.star.vturb.to_value(u.km / u.s)
        
        sme.abund = self.abund
        sme.linelist = ValdFile(self.linelist)

        sme.atmo.source = self.atmosphere
        sme.atmo.method = "grid"

        if self.nlte is not None:
            for elem, grid in self.nlte.items():
                sme.nlte.set_nlte(elem, grid)

        sme.wave = wave.to_value(u.AA)

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
        sme.teff = self.star.teff.to_value(u.K)
        sme.logg = self.star.logg.to_value(u.one)
        sme.monh = self.star.monh.to_value(u.one)
        
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
