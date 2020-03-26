from glob import glob
from os.path import dirname, join

import numpy as np
from tqdm import tqdm

from cats.spectrum import SpectrumArray, SpectrumList


def collect_observations(files):
    print("Loading data...")
    files = glob(files)
    spectra = [SpectrumList.read(f) for f in tqdm(files)]

    print("Sort observations by time")
    times = [spec.datetime for spec in spectra]
    sort = np.argsort(times)
    spectra = [spectra[i] for i in sort]
    times = [times[i] for i in sort]

    spectra = SpectrumArray(spectra)
    return spectra


if __name__ == "__main__":
    data_dir = join(dirname(__file__), "noise_1", "raw")
    target_dir = join(dirname(__file__), "noise_1", "medium")
    files = join(data_dir, "*.fits")

    spectra = collect_observations(files)

    fname = join(target_dir, "spectra.npz")
    print(f"Saving data to {fname}")
    spectra.write(fname)
    print("Done")
