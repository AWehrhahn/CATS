import os.path
import sys
import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np

from cats.cats import main
from cats import cats
from cats.data_modules.aronson import aronson

if len(sys.argv) > 1:
    parser = argparse.ArgumentParser(
        description="Extract the planetary transmittance spectrum, from one or more transit observations."
    )
    parser.add_argument("star", type=str, help="The observed star")
    parser.add_argument(
        "planet",
        type=str,
        help="The letter of the planet (default=b)",
        nargs="?",
        default="b",
    )
    parser.add_argument(
        "-l",
        "--lambda",
        type=str,
        help="Regularization parameter lambda (default=auto)",
        default="auto",
        dest="lamb",
    )

    args = parser.parse_args()
    star = args.star
    planet = args.planet
    lamb = args.lamb
    if lamb != "auto":
        try:
            lamb = float(lamb)
        except ValueError:
            logging.error("Invalid value for -l/-lambda")
            exit()
else:
    star = "GJ1214"
    planet = "b"
    lamb = 6810

try:
    # main(star, planet, lamb=lamb)
    
    # Generate synthetic data
    configuration = cats.load_configuration(star, planet)
    data = cats.load_data(star, planet, configuration)

    # Assume that the first guess for the planet size is off by 5%
    print(data["parameters"]["r_planet"])
    data["parameters"]["r_planet"] *= 1.05

    # Try to use Aronson method on it
    module = aronson(configuration)
    intensity = module.get_intensities(**data)

except FileNotFoundError as fnfe:
    logging.error("Some files seem to be missing, can't complete calculation")
    logging.error(fnfe)
