import logging
from copy import copy

import astropy.constants as const
import astropy.units as u
import numpy as np
import specutils
import specutils.manipulation as specman
from astropy.time import Time

from . import reference_frame as rf

logger = logging.getLogger(__name__)


class Spectrum1D(specutils.Spectrum1D):
    """
    Extends the specutils Spectrum1D class
    with a few convenience functions useful for
    our case.
    Most importently for resampling and shifting
    the spectrum to different wavelengths grids
    """

    reference_frame_values = ["barycentric", "telescope", "planet", "star"]

    def __init__(self, *args, **kwargs):
        # Set default options (if not given)
        kwargs["radial_velocity"] = kwargs.get("radial_velocity", 0 * (u.km / u.s))

        super().__init__(*args, **kwargs)

        # Which data source was this obtained from
        self.meta["source"] = ""
        # What does this data represent (e.g. stellar spectrum)
        self.meta["description"] = ""
        # Citation data (in bibtex format) if any
        self.meta["citation"] = ""
        # Additional data for reference frame changes
        self.meta["star"] = kwargs.get("star")
        self.meta["planet"] = kwargs.get("planet")
        self.meta["observatory_location"] = kwargs.get("observatory_location")
        self.meta["sky_location"] = kwargs.get("sky_location")
        # Datetime of the observation
        self.datetime = kwargs.get("datetime", Time(0, format="mjd"))
        # One of "barycentric", "telescope", "planet", "star"
        self.reference_frame = self.reference_frame_from_name(
            kwargs.get("reference_frame", "barycentric")
        )

    @property
    def datetime(self):
        return self.meta["datetime"]

    @datetime.setter
    def datetime(self, value):
        if not isinstance(value, Time):
            value = Time(value)
        self.meta["datetime"] = value

    @property
    def reference_frame(self):
        return self.meta["reference_frame"]

    @reference_frame.setter
    def reference_frame(self, value):
        if value not in self.reference_frame_values:
            raise ValueError(
                f"Reference frame not understood."
                f"Expected one of {self.reference_frame_values} but got {value}"
            )
        # TODO: conversion to ReferenceFrame object
        self.meta["reference_frame"] = value

    def reference_frame_from_name(self, frame):
        if frame == "barycentric":
            frame = rf.BarycentricFrame()
        elif frame == "telescope":
            frame = rf.TelescopeFrame(
                self.datetime,
                self.meta["observatory_location"],
                self.meta["sky_location"],
            )
        elif frame == "star":
            frame = rf.StarFrame(self.meta["star"])
        elif frame == "planet":
            frame = rf.PlanetFrame(
                self.meta["datetime"], self.meta["star"], self.meta["planet"]
            )
        else:
            raise ValueError(
                "Could not recognize reference frame name."
                f"Expected one of {self.reference_frame_values} but got {frame} instead."
            )
        return frame

    def shift(self, target_frame, inplace=False):
        """
        Shift the spectrum from the current
        to a new reference frame
        
        Acceptable reference frames are:
          - barycentric
          - telescope (NOT barycentric)
          - planet
          - star

        Parameters
        ----------
        target_frame : str
            the NEW reference frame
        inplace : bool, optional
            whether to perform the data shift in place
            (overriding existing data), or not (default: False)

        Returns
        -------
        spec: Spectrum1D
            the shifted spectrum structure
        """

        # TODO: conversion to/from star/planet requires info about the star/planet
        # TODO: don't forget to use the radial velocity as well (?)
        try:
            target_frame = self.reference_frame_from_name(target_frame)
        except ValueError:
            # Its already a frame ?
            pass

        # Step 1: determine relative velocity between current and target frame
        rv = self.reference_frame.to_frame(target_frame)

        # Step 2: Use the determined radial velocity to calculate a new wavelength grid
        shifted = copy(self.wavelength)
        beta = rv / const.c
        shifted *= np.sqrt((1 + beta) / (1 - beta))

        # Step 3: Create new Spectrum1D with shifted wavelength grid
        spec = Spectrum1D(spectral_axis=shifted, flux=self.flux, **self.meta)
        spec.reference_frame = target_frame

        return spec

    def resample(self, grid, method="flux_conserving", **kwargs):
        """
        Resample the current spectrum to a different wavelength grid

        Parameters
        ----------
        grid : Quantity
            The new wavelength grid
        method : str, optional
            The method to use for interpolation. Must be one of "flux_conserving", "linear", or "spline".
            By default "flux_conserving"
        ** kwargs
            Any optional keywords are passed on to the resampler. See specutils.manipulation for details

        Returns
        -------
        spec : Spectrum1D
            The resampled spectrum
        """
        if method == "flux_conserving":
            resampler = specman.FluxConservingResampler(**kwargs)
        elif method == "linear":
            resampler = specman.LinearInterpolatedResampler(**kwargs)
        elif method == "spline":
            resampler = specman.SplineInterpolatedResampler(**kwargs)

        spec = resampler(self, grid)

        # Cast spec to Spectrum 1D class and set meta parameters
        spec.meta = copy(self.meta)
        spec.radial_velocity = self.radial_velocity
        spec.with_velocity_convention(self.velocity_convention)
        spec.__class__ = Spectrum1D

        return spec
