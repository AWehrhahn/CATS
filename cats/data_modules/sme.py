import numpy as np
import astropy.units as u

from copy import deepcopy
from tqdm import tqdm

from .datasource import DataSource, StellarIntensities
from ..spectrum import Spectrum1D, SpectrumList

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


class SmeBase(DataSource):
    def __init__(
        self,
        star,
        abundance="solar",
        linelist=None,
        atmosphere="marcs",
        nlte=None,
        normalize=False,
    ):
        DataSource.__init__(self, "sme")

        self.star = star
        self.abund = abundance
        self.linelist = linelist
        self.atmosphere = atmosphere
        self.nlte = nlte
        self.normalize = normalize

        if self.atmosphere == "marcs":
            self.atmosphere = "marcs2014.sav"
            # vturb = self.star.vturb.to_value(u.km / u.s)
            # # round to nearest option
            # vturb = round_to_nearest(vturb, [1, 2, 5])
            # self.atmosphere = f"marcs2012p_t{vturb:1.1f}.sav"

    def synthesize(self, wrange, mu=None, intensities=False):
        sme = SME_Structure()
        # TODO other stellar parameters
        sme.teff = self.star.teff.to_value(u.K)
        sme.logg = self.star.logg.value
        sme.monh = self.star.monh
        sme.vturb = self.star.vturb.to_value(u.km / u.s)

        if mu is not None:
            sme.mu = mu

        sme.abund = self.abund
        sme.linelist = ValdFile(self.linelist)

        sme.atmo.source = self.atmosphere
        sme.atmo.method = "grid"

        if self.nlte is not None:
            for elem, grid in self.nlte.items():
                sme.nlte.set_nlte(elem, grid)

        sme.wran = [[wr.lower.to_value(u.AA), wr.upper.to_value(u.AA)] for wr in wrange]

        sme.cscale_flag = "none"
        sme.normalize_by_continuum = self.normalize
        sme.vrad_flag = "none"

        synthesizer = Synthesizer()
        if not intensities:
            sme = synthesizer.synthesize_spectrum(sme)
            wave, spec = sme.wave, sme.synth
        else:
            sme.specific_intensities_only = True
            wave, spec, cont = synthesizer.synthesize_spectrum(sme)

        wave = [w << u.AA for w in wave]
        if self.normalize:
            spec = [s << u.one for s in spec]
        else:
            spec = [s << flux_units for s in spec]

        return wave, spec, sme.citation("bibtex")


class SmeStellar(SmeBase):
    def __init__(
        self,
        star,
        abundance="solar",
        linelist=None,
        atmosphere="marcs",
        nlte=None,
        normalize=False,
    ):
        super().__init__(
            star,
            abundance=abundance,
            linelist=linelist,
            atmosphere=atmosphere,
            nlte=nlte,
            normalize=normalize,
        )
        self.is_prepared = False
        self.spectrum = None

    def prepare(self, wrange):
        wave, spec, citation = self.synthesize(wrange, mu=None, intensities=False)

        synth = SpectrumList(
            flux=spec,
            spectral_axis=wave,
            reference_frame="star",
            star=self.star,
            source="sme",
            description="synthetic stellar spectrum",
            citation=citation,
        )
        self.spectrum = synth
        self.is_prepared = True

    def get(self, wrange, time):
        if not self.is_prepared:
            self.prepare(wrange)

        synth = deepcopy(self.spectrum)
        for s in synth:
            s.datetime = time

        return synth


class SmeIntensities(SmeBase, StellarIntensities):
    # TODO: Rossiter-McLaughlin effect should blue/red shift the observations depending on the
    # rotation velocity of the star
    def __init__(
        self,
        star,
        planet,
        abundance="solar",
        linelist=None,
        atmosphere="marcs",
        nlte=None,
        normalize=False,
    ):
        SmeBase.__init__(
            self, star, abundance, linelist, atmosphere, nlte, normalize=normalize
        )
        StellarIntensities.__init__(self, star, planet)
        self.is_prepared = False
        self.times = None
        self.spectra = None

    def prepare(self, wrange, times):

        # Oversample mu by a factor of 5
        # that way we can estimate the average intensity
        d = self.orbit.projected_radius(times)
        r = self.orbit.planet.radius
        R = self.orbit.star.radius

        spectra = []
        subsampling = 5
        steps = np.linspace(-1, 1, subsampling)
        for i in tqdm(range(len(times))):
            time = times[i]
            dp = d[i] + steps * r
            mu = np.sqrt(1 - dp ** 2 / R ** 2)
            mu = mu[np.isfinite(mu)]

            if len(mu) != 0:
                wave, spec, citation = self.synthesize(wrange, mu, intensities=True)

                flux = [np.zeros(len(w)) << u.one for w in wave]
                for j in range(len(spec)):
                    flux[j] += np.mean(spec[j], axis=0)
            else:
                regions = [
                    [wr.lower.to_value(u.AA), wr.upper.to_value(u.AA)] for wr in wrange
                ]
                wave = [np.linspace(wmin, wmax, 100) << u.AA for wmin, wmax in regions]
                if self.normalize:
                    flux = [np.zeros(100) << u.one for _ in wrange]
                else:
                    flux = [np.zeros(100) << flux_units for _ in wrange]
                citation = ""

            synth = SpectrumList(
                flux=flux,
                spectral_axis=wave,
                reference_frame="star",
                datetime=time,
                star=self.star,
                source="sme",
                description="synthetic specific intensities",
                citation=citation,
            )
            spectra += [synth]

        self.times = times
        self.spectra = spectra
        self.is_prepared = True

    def get(self, wrange, time, mode="core"):
        if self.is_prepared and time in self.times:
            idx = [i for i, t in enumerate(self.times) if t == time][0]
            synth = deepcopy(self.spectra[idx])
        else:
            mu = self.orbit.mu(time)
            mu = np.atleast_1d(mu)
            wave, spec, citation = self.synthesize(wrange, mu, intensities=True)
            synth = wave, spec

        # TODO mode differences?
        # Integrate over a range of mu values for each given mu value?

        return synth
