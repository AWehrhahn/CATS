import os.path
import sys
import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np

from cats.cats import main
from cats import cats
from cats.data_modules.aronson import aronson

logger = logging.getLogger(__name__)


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
            logger.error("Invalid value for -l/-lambda")
            exit()
else:
    star = "GJ1214"
    planet = "b"
    lamb = "auto"

try:
    main(star, planet, lamb=lamb)

except FileNotFoundError as fnfe:
    logger.error("Some files seem to be missing, can't complete calculation")
    logger.error(fnfe)
