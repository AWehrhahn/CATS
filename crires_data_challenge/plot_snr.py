import logging
from os.path import dirname, join

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage import transform as tf

def shear(x, shear=1, inplace=False):
    afine_tf = tf.AffineTransform(shear=shear)
    modified = tf.warp(x, inverse_map=afine_tf)
    return modified

ranges = {
    50: (45, 96),
    100: (10,27),
    200: (2, 8)
}

shears = {
    50: -0.14,
    100: -0.4,
    200: -0.8
}

n_sysrem = 5
snr = 100

base_dir = join(dirname(__file__), f"../datasets/WASP-107b_SNR{snr}")
fname = join(base_dir, "medium", "cross_correlation.npz")

data = np.load(fname)
plt.imshow(data[f"{n_sysrem}"], aspect="auto")
plt.show()


for snr in [50, 100, 200]:
    base_dir = join(dirname(__file__), f"../datasets/WASP-107b_SNR{snr}")
    fname = join(base_dir, "medium", "cross_correlation.npz")
    data = np.load(fname)
    low, high = ranges[snr]
    d = data[f"{n_sysrem}"]
    # std = np.median(np.abs(d - np.median(d))) * 1.4826
    d = np.sum(shear(d[low:high], shears[snr]), axis=0)
    d -= np.median(d)
    d /= np.median(np.abs(d)) * 1.4826
    plt.plot(d, label=f"SNR {snr}")

plt.title(f"CCF with H2O Atmosphere\nSYSREM Iterations: {n_sysrem}")

plt.xlabel("v [km/s]")
xticks = plt.xticks()[0][1:-1]
xticks_labels = xticks - 100
plt.xticks(xticks, labels=xticks_labels)

plt.ylabel("ccf [SNR]")

plt.legend()
plt.show()