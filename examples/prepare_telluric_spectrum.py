from os.path import dirname, join

from tqdm import tqdm

from cats.simulator.detector import Crires
from cats.spectrum import SpectrumArray
from exoorbit.bodies import Star
from cats.extractor.prepare import create_telluric


if __name__ == "__main__":
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

    print("Save data...")
    stellar.write(join(target_dir, "telluric.npz"))
