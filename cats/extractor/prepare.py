from os.path import dirname, join

from tqdm import tqdm

from ..data_modules.sme import SmeStellar, SmeIntensities
from ..data_modules.telluric_model import TelluricModel
from ..simulator.detector import Crires
from ..spectrum import SpectrumArray


def create_stellar(wrange, spectra, star, times, linelist):
    print("Creating stellar...")
    stellar = SmeStellar(star, linelist=linelist, normalize=True)
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


def create_intensities(wrange, spectra, star, planet, observatory, times, linelist):
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


def create_telluric(wrange, spectra, star, observatory, times):
    print("Creating tellurics...")
    telluric = TelluricModel(star, observatory)
    reference_frame = spectra.reference_frame
    result = []
    for i, time in tqdm(enumerate(times), total=len(times)):
        wave = spectra[i].wavelength
        spec = telluric.get(wrange, time)
        spec = spec.shift(reference_frame, inplace=True)
        spec = spec.resample(wave, method="linear")
        result += [spec]
    result = SpectrumArray(result)
    return result
