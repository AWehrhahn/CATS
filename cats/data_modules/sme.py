import numpy as np
import astropy.units as u

from .datasource import DataSource, StellarIntensities
from ..spectrum import Spectrum1D

from pysme.util import start_logging
from pysme.sme import SME_Structure
from pysme.synthesize import Synthesizer
from pysme.linelist.vald import ValdFile

# start_logging()

flux_units = u.erg / u.cm ** 2 / u.s / u.AA


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
    def __init__(
        self, star, abundance="solar", linelist=None, atmosphere="marcs", nlte=None
    ):
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

    def get(self, wave, time):
        sme = SME_Structure()
        # TODO other stellar parameters
        sme.teff = self.star.teff.to_value(u.K)
        sme.logg = self.star.logg.to_value(u.one)
        sme.monh = self.star.monh.to_value(u.dex)
        sme.vturb = self.star.vturb.to_value(u.km / u.s)

        sme.abund = self.abund
        sme.linelist = ValdFile(self.linelist)

        sme.atmo.source = self.atmosphere
        sme.atmo.method = "grid"

        if self.nlte is not None:
            for elem, grid in self.nlte.items():
                sme.nlte.set_nlte(elem, grid)

        w = wave.to_value(u.AA)
        wmin, wmax = w.min(), w.max()
        sme.wran = [wmin, wmax]

        sme.cscale_flag = "none"
        sme.normalize_by_continuum = False
        sme.vrad_flag = "none"

        synthesizer = Synthesizer()
        sme = synthesizer.synthesize_spectrum(sme)

        wave = sme.wave[0] << u.AA
        spec = sme.synth[0] << flux_units

        synth = Spectrum1D(
            flux=spec,
            spectral_axis=wave,
            reference_frame="barycentric",
            datetime=time,
            star=self.star,
            source="sme",
            description="synthetic stellar spectrum",
            citation=sme.citation(output="bibtex"),
        )
        return synth


class SmeIntensities(StellarIntensities):
    def __init__(
        self,
        star,
        planet,
        abundance="solar",
        linelist=None,
        atmosphere="marcs",
        nlte=None,
    ):
        super().__init__(star, planet)

        self.abund = abundance
        self.linelist = linelist
        self.atmosphere = atmosphere
        self.nlte = nlte

        if self.atmosphere == "marcs":
            vturb = self.star.vturb.to_value(u.km / u.s)
            # round to nearest option
            vturb = round_to_nearest(vturb, [0, 1, 2, 3, 4, 5])
            self.atmosphere = f"marcs2012p_t{vturb:1.1f}.sav"

    def get(self, wave, time, mode="core"):
        mu = self.orbit.phase_angle(time)
        mu = np.atleast_1d(mu)

        mask = mu > 0

        sme = SME_Structure()
        sme.teff = self.star.teff.to_value(u.K)
        sme.logg = self.star.logg.to_value(u.one)
        sme.monh = self.star.monh.to_value(u.dex)
        sme.vturb = self.star.vturb.to_value(u.km / u.s)

        sme.mu = mu[mask]

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

        sme.specific_intensities_only = True

        synthesizer = Synthesizer()
        wave, spec, cont = synthesizer.synthesize_spectrum(sme)

        wave = wave << u.AA
        spec = spec << flux_units

        spec_full = np.zeros((mu.size, spec.shape[-1])) << flux_units
        spec_full[mask] = spec[0]

        synth = Spectrum1D(
            flux=spec_full,
            spectral_axis=wave[0],
            reference_frame="barycentric",
            star=self.star,
            planet=self.planet,
            datetime=time,
            source="sme",
            description="stellar specific intensities",
            citation=sme.citation(output="bibtex"),
        )

        # TODO mode differences?
        # Integrate over a range of mu values for each given mu value?

        return synth
