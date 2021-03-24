# The following is an algorithm to remove systematic effects in a large set of
# light curves based on a paper by O. Tamuz, T. Mazeh and S. Zucker (2004)
# titled "Correcting systematic effects in a large set of photometric light
# curves".
from __future__ import print_function, division
import sys
import os.path
import os
import warnings

import astropy.io.ascii as at
import numpy as np
from tqdm import tqdm

from ..spectrum import SpectrumBase


# Set the output print options:
# np.set_printoptions(threshold=np.nan, precision=6)


def generate_matrix(spectra, errors=None):

    # print("generating residuals matrix")
    epoch_dim, stars_dim = spectra.shape
    # Calculate the median along the epoch axis
    if isinstance(spectra, SpectrumBase):
        flux = spectra.flux.to_value(1)
        if errors is None and spectra.uncertainty is not None:
            errors = spectra.uncertainty.array
    else:
        flux = spectra

    median_list = np.nanmedian(flux, axis=0)
    # Calculate residuals from the ORIGINAL light curve
    residuals = (flux - median_list).T
    # For the data points with quality flags != 0,
    # set the errors to a large value

    if errors is None:
        errors = np.sqrt(np.abs(flux)).T
    else:
        errors = errors.T

    # print("finished matrix")

    return residuals, errors, median_list, flux


def sysrem(input_star_list, num_errors=5, iterations=10, errors=None):

    residuals, errors, median_list, star_list = generate_matrix(input_star_list, errors)
    stars_dim, epoch_dim = residuals.shape

    # print("starting sysrem")

    # This medians.txt file is a 2D list with the first column being the medians
    # of stars' magnitudes at different epochs (the good ones) and their
    # standard deviations, so that they can be plotted against the results after
    # errors are taken out below.
    for n in tqdm(
        range(num_errors), desc="Removing Systematic #", leave=False
    ):  # The number of linear systematics to remove
        c = np.zeros(stars_dim)
        a = np.ones(epoch_dim)

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=RuntimeWarning)
            # minimize a and c values for a number of iterations, iter
            for i in range(iterations):
                # Using the initial guesses for each a value of each epoch, minimize c for each star
                err_squared = errors ** 2
                numerator = np.nansum(a * residuals / err_squared, axis=1)
                denominator = np.nansum(a ** 2 / err_squared, axis=1)
                c = numerator / denominator

                # Using the c values found above, minimize a for each epoch
                numerator = np.nansum(c[:, None] * residuals / err_squared, axis=0)
                denominator = np.nansum(c[:, None] ** 2 / err_squared, axis=0)
                a = numerator / denominator

        # Create a matrix for the systematic errors:
        # syserr = np.zeros((stars_dim, epoch_dim))
        syserr = c[:, None] * a[None, :]

        # Remove the systematic error
        residuals = residuals - syserr

    # Reproduce the results in terms of medians and standard deviations for plotting
    # corrected_mags = residuals + median_list[:, None]
    # final_median = np.median(corrected_mags, axis=1)
    # std_dev = np.std(corrected_mags, axis=1)

    return residuals.T
