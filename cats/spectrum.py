import logging
from copy import copy
from collections import Sequence

import operator as op

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

        meta = {}
        # Which data source was this obtained from
        meta["source"] = kwargs.pop("source", "")
        # What does this data represent (e.g. stellar spectrum)
        meta["description"] = kwargs.pop("description", "")
        # Citation data (in bibtex format) if any
        meta["citation"] = kwargs.pop("citation", "")
        # Additional data for reference frame changes
        meta["star"] = kwargs.pop("star", None)
        meta["planet"] = kwargs.pop("planet", None)
        meta["observatory_location"] = kwargs.pop("observatory_location", None)
        meta["sky_location"] = kwargs.pop("sky_location", None)
        # Datetime of the observation
        meta["datetime"] = kwargs.pop("datetime", Time(0, format="mjd"))
        # One of "barycentric", "telescope", "planet", "star"
        reference_frame = kwargs.pop("reference_frame", "barycentric")

        kwmeta = kwargs.get("meta", {}) if kwargs.get("meta") is not None else {}
        kwmeta.update(meta)
        kwargs["meta"] = kwmeta

        super().__init__(*args, **kwargs)

        self.reference_frame = reference_frame

    def __len__(self):
        return len(self.spectral_axis)

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
        if (
            not isinstance(value, rf.ReferenceFrame)
            and value not in self.reference_frame_values
        ):
            raise ValueError(
                f"Reference frame not understood."
                f"Expected one of {self.reference_frame_values} but got {value}"
            )
        if value in self.reference_frame_values:
            value = self.reference_frame_from_name(value)
        self.meta["reference_frame"] = value

    def reference_frame_from_name(self, frame):
        if frame == "barycentric":
            frame = rf.BarycentricFrame()
        elif frame == "telescope":
            frame = rf.TelescopeFrame(
                self.datetime,
                self.meta["observatory_location"],
                sky_location=(self.meta["star"].ra, self.meta["star"].dec),
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
        options = ["flux_conserving", "linear", "spline"]
        if method == "flux_conserving":
            resampler = specman.FluxConservingResampler(**kwargs)
        elif method == "linear":
            resampler = specman.LinearInterpolatedResampler(**kwargs)
        elif method == "spline":
            resampler = specman.SplineInterpolatedResampler(**kwargs)
        else:
            raise ValueError(f"Interpolation method not understood. Expected one of {options}, but got {method}")

        spec = resampler(self, grid)

        # Cast spec to Spectrum 1D class and set meta parameters
        spec.meta = copy(self.meta)
        spec.radial_velocity = self.radial_velocity
        spec.with_velocity_convention(self.velocity_convention)
        spec.__class__ = Spectrum1D

        return spec

class SpectrumList(Sequence):
    """
    Stores a list of Spectrum1D objects, with shared metadata
    This usually represents the different orders of the spectrum,
    which may have various sizes of spectral axis, especially when
    using model data

    Really this is only a thin shell with convenience functions,
    to act on all spectra at once

    NOTE
    ----
    No checks whatsoever are performed on the input data. Please
    only use CATS Spectra1D objects as input

    """
    def __init__(self, flux, spectral_axis, **kwargs):
        super().__init__()

        # We actually just pass everything to each individual spectrum
        # instead of trying to organize the metadata into one place
        # This means we don't have to worry about setting the values later
        # But this also means that they could change if we are not careful
        self._data = []
        for f, sa in zip(flux, spectral_axis):
            spec = Spectrum1D(flux=f, spectral_axis=sa, **kwargs)
            self._data += [spec]

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        # TODO: check that its a Spectrum1D ?
        self._data[key] = value

    def __len__(self):
        return len(self._data)

    def __operator__(self, other, operator):
        if isinstance(other, (float, int)) or (hasattr(other, "size") and other.size == 1):
            # If its scalar, make it an array of the same length
            other = [other for _ in self]
        elif len(other) != len(self):
            raise ValueError(f"Incompatible sizes of {len(self)} and {len(other)}")
        data = [operator(t, o) for t, o in zip(self, other)]
        sl = self.__class__.from_spectra(data)
        return sl

    def __add__(self, other):
        return self.__operator__(other, op.add)

    def __sub__(self, other):
        return self.__operator__(other, op.sub)

    def __mul__(self, other):
        return self.__operator__(other, op.mul)

    def __truediv__(self, other):
        return self.__operator__(other, op.truediv)

    @property
    def shape(self):
        return (len(self), [len(d) for d in self])

    @property
    def size(self):
        return sum(self.shape[1])

    @classmethod
    def from_spectra(cls, spectra):
        """
        Create a new SpectrumList object from a list of Spectrum1D
        
        Note
        ----
        The input spectra are NOT copied, therefore any changes
        made will be present in the existing variables

        Parameters
        ----------
        spectra : list
            list of Spectrum1D
        
        Returns
        -------
        spectrum_list : SpectrumList
            SpectrumList with containing the spectra
        """
        spectrum_list = cls([], [])
        spectrum_list._data = spectra
        return spectrum_list


    def resample(self, grid, **kwargs):        
        """
        Resample the different spectra onto the given grid
        
        Parameters
        ----------
        grid : list
            list of wavelengths arrays to resample to
        ** kwargs
            keyword arguments passed to Spectrum1D resample

        Returns
        -------
        SpectrumList
            New resampled SpectrumList object
        """

        if len(grid) != len(self):
            raise ValueError(f"Wavelength grid has {len(grid)} arrays, but SpectrumList has {len(self)} spectra.")

        data = []
        for i, (g, spec) in enumerate(zip(grid, self._data)):
            s = spec.resample(g, **kwargs)
            data += [s]

        sl = SpectrumList.from_spectra(data)
        return sl

    def shift(self, target_frame, **kwargs):
        """
        Shift all spectra to the target frame
        
        Parameters
        ----------
        target_frame : ReferenceFrame
            target reference frame to convert the Spectra to
        ** kwargs
            keyword arguments passed to Spectrum1D shift
        Returns
        -------
        SpectrumList
            New shifted SpectrumList object
        """

        data = []
        for i, spec in enumerate(self._data):
            s = spec.shift(target_frame, **kwargs)
            data += [s]

        sl = SpectrumList.from_spectra(data)
        return sl
