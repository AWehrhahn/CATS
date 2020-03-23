import logging
from copy import copy, deepcopy
from collections import Sequence
from os import makedirs
from os.path import dirname, abspath
import operator as op

from datetime import datetime

import astropy.constants as const
import astropy.units as u
from astropy import coordinates as coords
import numpy as np
import specutils
import specutils.manipulation as specman
from astropy.time import Time
from astropy.io import fits

from . import reference_frame as rf
from .simulator import detector
from .data_modules.stellar_db import StellarDb

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

    def __copy__(self):
        cls = self.__class__
        wave = np.copy(self.wavelength)
        flux = np.copy(self.flux)
        meta = self.meta

        other = cls(flux=flux, spectral_axis=wave, **meta)
        return other

    def __deepcopy__(self, *args, **kwargs):
        cls = self.__class__
        wave = np.copy(self.wavelength)
        flux = np.copy(self.flux)
        meta = deepcopy(self.meta)

        other = cls(flux=flux, spectral_axis=wave, **meta)
        return other

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

    @property
    def regions(self):
        wmin, wmax = self.wavelength[[0, -1]]
        return specutils.SpectralRegion(wmin, wmax)

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
                self.meta["observatory_location"],
                sky_location=self.meta["star"].coordinates,
            )
        elif frame == "star":
            frame = rf.StarFrame(self.meta["star"])
        elif frame == "planet":
            frame = rf.PlanetFrame(self.meta["star"], self.meta["planet"])
        else:
            raise ValueError(
                "Could not recognize reference frame name."
                f"Expected one of {self.reference_frame_values} but got {frame} instead."
            )
        return frame

    def _get_fits_hdu(self):
        wave = self.wavelength.to(u.AA)
        flux = self.flux.decompose()
        wave_unit = str(wave.unit)
        flux_unit = str(flux.unit)
        wave = fits.Column(
            name="wavelength", array=wave.value, format="D", coord_unit=wave_unit
        )
        flux = fits.Column(
            name="flux", array=flux.value, format="D", coord_unit=flux_unit
        )

        header = {
            "SOURCE": self.meta["source"],
            "DESCR": self.meta["description"],
            "CITATION": self.meta["citation"].replace("\n", "").strip(),
            "DATE-OBS": self.datetime.fits,
            "DATE": datetime.now().isoformat(),
            "REFFRAME": str(self.reference_frame),
        }

        # This only saves the name
        # Values have to be recovered using various sources
        # StellarDB for star and planet
        # EarthCoordinates for observatory
        if self.meta["star"] is not None:
            header["STAR"] = str(self.meta["star"])
        if self.meta["planet"] is not None:
            header["PLANET"] = str(self.meta["planet"])
        if self.meta["observatory_location"] is not None:
            if isinstance(self.meta["observatory_location"], str):
                header["OBSNAME"] = self.meta["observatory_location"]
            elif isinstance(self.meta["observatory_location"], (tuple, list)):
                header["OBSLAT"] = (self.meta["observatory_location"][0],)
                header["OBSLON"] = (self.meta["observatory_location"][1],)
                header["OBSALT"] = (self.meta["observatory_location"][2],)
        if self.meta["sky_location"] is not None:
            if isinstance(self.meta["sky_location"], coords.SkyCoord):
                header["RA"] = self.meta["sky_location"].ra.to_value("hourangle")
                header["DEC"] = self.meta["sky_location"].dec.to_value("deg")
            else:
                header["RA"] = self.meta["sky_location"][0].to_value("hourangle")
                header["DEC"] = self.meta["sky_location"][1].to_value("deg")

        header = fits.Header(header)
        hdu = fits.BinTableHDU.from_columns([wave, flux], header=header)
        return hdu

    @classmethod
    def _read_fits_hdu(cls, hdu):
        header = hdu.header
        data = hdu.data
        meta = {}

        wunit = data.columns["wavelength"].coord_unit
        wunit = u.Unit(wunit) if wunit is not None else u.one
        funit = data.columns["flux"].coord_unit
        funit = u.Unit(funit) if funit is not None else u.one
        wave = data["wavelength"] << wunit
        flux = data["flux"] << funit

        meta["source"] = header["SOURCE"]
        meta["description"] = header["DESCR"]
        meta["citation"] = header["CITATION"]
        meta["datetime"] = Time(header["DATE-OBS"], format="fits")
        meta["reference_frame"] = header["REFFRAME"]

        sdb = StellarDb()
        if "STAR" in header:
            star = header["STAR"]
            star = sdb.get(star)
            meta["star"] = star
        if "PLANET" in header:
            planet = header["PLANET"]
            planet = star.planets[planet]
            meta["planet"] = planet
        if "OBSNAME" in header:
            meta["observatory_location"] = header["OBSNAME"]
        elif "OBSLON" in header:
            meta["observatory_location"] = (
                header["OBSLON"],
                header["OBSLAT"],
                header["OBSALT"],
            )
        if "RA" in header:
            ra = coords.Angle(header["RA"], "hourangle")
            dec = header["DEC"] * u.deg
            meta["sky_location"] = ra, dec

        spec = cls(flux=flux, spectral_axis=wave, **meta)

        return spec

    def write(self, fname):
        hdu = self._get_fits_hdu()
        hdu.writeto(fname, overwrite=True)

    @staticmethod
    def read(fname):
        hdulist = fits.open(fname)
        spec = Spectrum1D._read_fits_hdu(hdulist[0])
        return spec

    def shift(self, target_frame, inplace=False, rv=None):
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
            pass

        # Step 1: determine relative velocity between current and target frame
        if rv is None:
            rv = self.reference_frame.to_frame(target_frame, self.datetime)

        # Step 2: Use the determined radial velocity to calculate a new wavelength grid
        if not inplace:
            shifted = np.copy(self.wavelength)
        else:
            shifted = self.wavelength
        beta = rv / const.c
        shifted *= np.sqrt((1 + beta) / (1 - beta))

        # Step 3: Create new Spectrum1D with shifted wavelength grid
        if inplace:
            self.spectral_axis = shifted
            self.reference_frame = target_frame
            spec = self
        else:
            spec = Spectrum1D(spectral_axis=shifted, flux=self.flux, **self.meta)
            spec.reference_frame = target_frame

        return spec

    def resample(self, grid, method="spline", **kwargs):
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
            raise ValueError(
                f"Interpolation method not understood. Expected one of {options}, but got {method}"
            )

        spec = resampler(self, grid)

        # Cast spec to Spectrum 1D class and set meta parameters
        spec.meta = copy(self.meta)
        spec.radial_velocity = self.radial_velocity
        spec.with_velocity_convention(self.velocity_convention)
        spec.__class__ = Spectrum1D

        return spec

    def extract_region(self, wrange):
        wave, flux = [], []
        for wmin, wmax in wrange.subregions:
            mask = (self.wavelength >= wmin) & (self.wavelength <= wmax)

            wave += [self.wavelength[mask]]
            flux += [self.flux[mask]]

        if len(wrange) == 1:
            spec = Spectrum1D(flux=flux[0], spectral_axis=wave[0], **self.meta)
        else:
            spec = SpectrumList(flux=flux, spectral_axis=wave, **self.meta)

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
        if isinstance(other, (float, int)) or (
            hasattr(other, "size") and other.size == 1
        ):
            # If its scalar, make it an array of the same length
            other = [other for _ in self]
        elif len(other) != len(self):
            raise ValueError(f"Incompatible sizes of {len(self)} and {len(other)}")
        # TODO: check that reference frame is the same
        data = [operator(t, o) for t, o in zip(self, other)]
        sl = self.__class__.from_spectra(data)
        for out, inp in zip(sl, self):
            out.meta = inp.meta
        return sl

    def __add__(self, other):
        return self.__operator__(other, op.add)

    def __sub__(self, other):
        return self.__operator__(other, op.sub)

    def __mul__(self, other):
        return self.__operator__(other, op.mul)

    def __truediv__(self, other):
        return self.__operator__(other, op.truediv)

    def __copy__(self):
        return SpectrumList.from_spectra(self._data)

    def __deepcopy__(self, *args, **kwargs):
        data = [deepcopy(s, *args, **kwargs) for s in self]
        return SpectrumList.from_spectra(data)

    @property
    def shape(self):
        return (len(self), [len(d) for d in self])

    @property
    def size(self):
        return sum(self.shape[1])

    @property
    def regions(self):
        wrange = [spec.regions for spec in self]
        for wr in wrange[1:]:
            wrange[0] += wr
        return wrange[0]

    @property
    def flux(self):
        return [s.flux for s in self]

    @property
    def wavelength(self):
        return [s.wavelength for s in self]

    @property
    def datetime(self):
        return self[0].datetime

    @datetime.setter
    def datetime(self, value):
        for s in self:
            s.datetime = value

    @property
    def reference_frame(self):
        return self[0].reference_frame

    @reference_frame.setter
    def reference_frame(self, value):
        for s in self:
            s.reference_frame = value

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

    def write(self, fname, detector=None):
        """Save all spectra in SpectrumList to a single fits file
        Each Spectrum1D has its own extension with a complete header
        
        Parameters
        ----------
        fname : str
            output filename
        detector : Detector, optional
            Detector used, will add some data to the primary header. By default None
        """
        # Save all in one fits file
        # with metatdata in the header and the wavelength and flux in the data

        header = {}
        if detector is not None:
            header["INSTRUME"] = str(detector)
            header["INS SET"] = detector.setting

        header = fits.Header(header)
        primary = fits.PrimaryHDU(header=header)
        secondary = []
        for spec in self:
            secondary += [spec._get_fits_hdu()]

        hdulist = fits.HDUList(hdus=[primary, *secondary])
        makedirs(dirname(abspath(fname)), exist_ok=True)
        hdulist.writeto(fname, overwrite=True)

    @classmethod
    def read(cls, fname):
        hdulist = fits.open(fname)
        header = hdulist[0].header

        det = None
        if "INSTRUME" in header:
            det = header["INSTRUME"]
            setting = header["INS SET"]
            det = det.capitalize().replace("+", "")

            module = getattr(detector, det)
            det = module(setting)

        spectra = []
        for i in range(1, len(hdulist)):
            spectra += [Spectrum1D._read_fits_hdu(hdulist[i])]
            if det is not None:
                spectra[-1].meta["observatory_location"] = det.observatory
                spectra[-1].reference_frame.observatory = det.observatory

        speclist = cls.from_spectra(spectra)
        return speclist

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
            raise ValueError(
                f"Wavelength grid has {len(grid)} arrays, but SpectrumList has {len(self)} spectra."
            )

        data = []
        for i, (g, spec) in enumerate(zip(grid, self)):
            s = spec.resample(g, **kwargs)
            data += [s]

        sl = SpectrumList.from_spectra(data)
        return sl

    def shift(self, target_frame, inplace=False, rv=None, **kwargs):
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

        try:
            target_frame = self[0].reference_frame_from_name(target_frame)
        except ValueError:
            pass

        if rv is None:
            rv = self.reference_frame.to_frame(target_frame, self.datetime)

        data = []
        for spec in self:
            s = spec.shift(target_frame, inplace=inplace, rv=rv, **kwargs)
            data += [s]

        if not inplace:
            sl = SpectrumList.from_spectra(data)
            return sl
        else:
            return self
