from os.path import dirname, join

from tqdm import tqdm

from cats.data_modules.sme import SmeStellar
from cats.simulator.detector import Crires
from cats.spectrum import SpectrumArray
from exoorbit.bodies import Star


def create_stellar(wrange, spectra, star, times):
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
print("Create stellar spectra...")
stellar = create_stellar(wrange, spectra, star, times)
stellar.write(join(target_dir, "stellar.npz"))
