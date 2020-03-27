from os.path import dirname, join

from tqdm import tqdm
from exoorbit.bodies import Star, Planet

from cats.simulator.detector import Crires
from cats.spectrum import SpectrumArray
from cats.extractor.prepare import create_intensities


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
    planet = Planet.load(join(target_dir, "planet.yaml"))

    # Create stellar data
    print("Create stellar intensities...")
    stellar = create_intensities(wrange, spectra, star, planet, observatory, times)

    print("Save data...")
    stellar.write(join(target_dir, "intensities.npz"))
