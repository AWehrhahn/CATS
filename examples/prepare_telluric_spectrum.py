from os.path import dirname, join

from tqdm import tqdm

from cats.data_modules.telluric_model import TelluricModel
from cats.simulator.detector import Crires
from cats.spectrum import SpectrumArray
from exoorbit.bodies import Star


def create_telluric(wrange, spectra, star, observatory, times):
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


target_dir = join(dirname(__file__), "noise_1", "medium")
util_dir = join(dirname(__file__), "noise_1")

detector = Crires("H/1/4", [1, 2, 3])
observatory = detector.observatory
wrange = detector.regions
linelist = join(util_dir, "crires_h_1_4.lin")

# Load data from disk
print("Load data...")
spectra = SpectrumArray.read(join(target_dir, "spectra.npz"))
times = spectra.datetime
star = Star.load(join(target_dir, "star.yaml"))


# Create stellar data
print("Create telluric data...")
stellar = create_telluric(wrange, spectra, star, observatory, times)
stellar.write(join(target_dir, "telluric.npz"))
