# Loads TAPAS tellurics

import glob
import gzip
from os.path import dirname, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm


def load_tellurics(files):
    telfil = glob.glob(files)  # reading the tellurics

    # Parse the header
    airmass, ang = np.zeros(np.size(telfil)), np.zeros(np.size(telfil))
    for i, ff in enumerate(telfil):
        with gzip.open(ff) as file:
            data = [file.readline().decode() for _ in range(23)]
            airmass[i] = np.float(data[15].strip()[9:])
            ang[i] = np.float(data[14].strip()[4:])

    # Parse the data
    tell = [None for _ in telfil]
    sort = np.argsort(airmass)
    airmass, ang = airmass[sort], ang[sort]
    for i in tqdm(range(len(airmass))):
        # Pandas is faster at parsing tables than numpy
        buff = pd.read_table(
            telfil[sort[i]],
            skiprows=23,
            header=None,
            names=["wavelength", "transmittance"],
            skipinitialspace=True,
            sep=r"\s+",
        )
        tell[i] = buff.values
    # Combine all the data in the end
    tell = np.stack(tell)

    print(ang)
    return tell, airmass, ang


tapas_dir = join(dirname(__file__), "../../data/tapas/")

tellw, airw, angw = load_tellurics(join(tapas_dir, "*winter*ipac.gz"))
tells, airs, angs = load_tellurics(join(tapas_dir, "*summer*ipac.gz"))

wavew, waves = np.squeeze(tellw[0, :, 0]), np.squeeze(tells[0, :, 0])
iiw, iis = np.argsort(wavew), np.argsort(waves)
tellw, tells = tellw[:, iiw, :], tells[:, iis, :]
tellwi = RegularGridInterpolator(
    (airw, np.squeeze(tellw[0, :, 0])), np.squeeze(tellw[:, :, 1])
)

pass
