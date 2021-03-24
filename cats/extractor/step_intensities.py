from tqdm import tqdm

from ..data_modules.sme import SmeIntensities
from ..spectrum import SpectrumArray, SpectrumArrayIO
from .steps import Step


class SpecificIntensitiesSmeStep(Step, SpectrumArrayIO):
    filename = "intensities.npz"

    def run(self, detector, spectra, star, planet, observatory, linelist):
        wrange = detector.regions
        intensities = self.create_intensities(
            wrange, spectra, star, planet, observatory, times, linelist
        )
        self.save(intensities)
        return intensities

    def create_intensities(
        self, wrange, spectra, star, planet, observatory, times, linelist
    ):
        print("Creating intensities...")

        stellar = SmeIntensities(star, planet, linelist=linelist, normalize=True)
        stellar.prepare(wrange, times)
        reference_frame = spectra.reference_frame
        result = []
        for i, time in tqdm(enumerate(times), total=len(times)):
            wave = spectra[i].wavelength
            spec = stellar.get(wrange, time)
            spec = spec.shift(reference_frame, inplace=True)
            spec = spec.resample(wave, method="linear")
            result += [spec]
        result = SpectrumArray(result)
        return result
