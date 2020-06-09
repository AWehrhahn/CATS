import logging
import operator as op
from collections import Sequence
from copy import copy, deepcopy
from datetime import datetime
from os import makedirs
from os.path import abspath, dirname

from tqdm import tqdm
import astropy.constants as const
import astropy.units as u
import numpy as np
import specutils
import specutils.manipulation as specman
from astropy import coordinates as coords
from astropy.io import fits
from astropy.time import Time

from . import reference_frame as rf
from .data_modules.stellar_db import StellarDb
from .simulator import detector

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

        if "spectral_axis" in kwargs.keys():
            pass
        else:
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
        # Datetime of the observation
        meta["datetime"] = kwargs.pop("datetime", Time(0, format="mjd"))
        # airmass is somewhat redundant (it can be calculated from star, observatory, and time)
        meta["airmass"] = kwargs.pop("airmass", None)
        # One of "barycentric", "telescope", "planet", "star"
        reference_frame = kwargs.pop("reference_frame", "barycentric")

        # Obsolete keywords
        kwargs.pop("sky_location", None)

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
        frame = rf.reference_frame_from_name(
            frame,
            star=self.meta["star"],
            planet=self.meta["planet"],
            observatory=self.meta["observatory_location"],
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
            header["RA"] = self.meta["star"].coordinates.ra.to_value("hourangle")
            header["DEC"] = self.meta["star"].coordinates.dec.to_value("deg")
        if self.meta["planet"] is not None:
            header["PLANET"] = str(self.meta["planet"])
        if self.meta["observatory_location"] is not None:
            if isinstance(self.meta["observatory_location"], str):
                header["OBSNAME"] = self.meta["observatory_location"]
            elif isinstance(self.meta["observatory_location"], (tuple, list)):
                header["OBSLAT"] = (self.meta["observatory_location"][0],)
                header["OBSLON"] = (self.meta["observatory_location"][1],)
                header["OBSALT"] = (self.meta["observatory_location"][2],)

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
        if "RA" in header and "DEC" in header:
            ra = coords.Angle(header["RA"], "hourangle")
            dec = header["DEC"] * u.deg
            meta["star"].coordinates = coords.SkyCoord(ra, dec)

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
            self._spectral_axis = shifted
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
            self._data = np.nan_to_num(self._data)
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
    def meta(self):
        return self[0].meta

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


class SpectrumArray(Sequence):
    """
    A Collection of Spectra with the same size,
    but possibly different wavelength axis
    """

    def __init__(self, *args, **kwargs):
        if len(args) == 1:
            if isinstance(args[0], list):
                spectra = args[0]
            elif isinstance(args[0], SpectrumList):
                spectra = [args[0]]
            else:
                raise ValueError

            nspec = len(spectra)
            nseg = spectra[0].shape[0]
            npix = spectra[0].size

            wunit = spectra[0][0].wavelength.unit
            funit = spectra[0][0].flux.unit

            self.wavelength = np.zeros((nspec, npix)) << wunit
            self.flux = np.zeros((nspec, npix)) << funit
            for i, spec in enumerate(spectra):
                self.wavelength[i] = np.concatenate(spec.wavelength)
                self.flux[i] = np.concatenate(spec.flux)

            self.segments = np.zeros(nseg + 1, dtype=int)
            self.segments[1:] = spectra[0].shape[1]
            self.segments = np.cumsum(self.segments)

            self.meta = spectra[0].meta
            times = Time([spec.datetime for spec in spectra])
            self.meta["datetime"] = times
        elif len(args) == 0 and len(kwargs) > 0:
            self.wavelength = kwargs.pop("spectral_axis")
            self.flux = kwargs.pop("flux")
            self.segments = kwargs.pop("segments")
            self.meta = kwargs
        else:
            raise ValueError

    def __len__(self):
        return len(self.wavelength)

    def __getitem__(self, key):
        wave = self.wavelength[key]
        flux = self.flux[key]

        spectra = []
        for left, right in zip(self.segments[:-1], self.segments[1:]):
            meta = self.meta.copy()
            meta["datetime"] = self.datetime[key]
            spec = Spectrum1D(
                flux=flux[left:right], spectral_axis=wave[left:right], **meta
            )
            spectra += [spec]

        speclist = SpectrumList.from_spectra(spectra)
        return speclist

    def __setitem__(self, key, value):
        self.wavelength[key] = np.concatenate(value.wavelength)
        self.flux[key] = np.concatenate(value.flux)

    @property
    def datetime(self):
        return self.meta["datetime"]

    @property
    def reference_frame(self):
        return self.meta["reference_frame"]

    @property
    def shape(self):
        return self.wavelength.shape

    def get_segment(self, seg):
        left, right = self.segments[seg : seg + 2]
        wave = self.wavelength[:, left:right]
        flux = self.flux[:, left:right]
        specarr = SpectrumArray(
            flux=flux, spectral_axis=wave, segments=[0, wave.shape[1]], **self.meta
        )
        return specarr

    def write(self, fname):
        np.savez(
            fname,
            wavelength=self.wavelength.value,
            flux=self.flux.value,
            meta=self.meta,
            segments=self.segments,
            flux_unit=str(self.flux.unit),
            wave_unit=str(self.wavelength.unit),
        )

    @classmethod
    def read(cls, fname):
        data = np.load(fname, allow_pickle=True)
        meta = data["meta"][()]
        flux_unit = data["flux_unit"][()]
        wave_unit = data["wave_unit"][()]
        flux = data["flux"] << u.Unit(flux_unit)
        wave = data["wavelength"] << u.Unit(wave_unit)
        segments = data["segments"]
        self = cls(flux=flux, spectral_axis=wave, segments=segments, **meta)
        return self

    def shift(self, target_frame, inplace=True):
        for i in tqdm(range(len(self))):
            meta = self.meta.copy()
            meta["datetime"] = self.datetime[i]
            spec = Spectrum1D(
                flux=self.flux[i], spectral_axis=self.wavelength[i], **meta
            )
            spec.shift(target_frame)
            self.wavelength[i] = spec.wavelength
            target_frame = spec.reference_frame

        self.meta["reference_frame"] = target_frame
        return self

    def resample(self, wavelength, inplace=True):
        for i in tqdm(range(len(self))):
            spec = Spectrum1D(flux=self.flux[i], spectral_axis=self.wavelength[i])
            spec = spec.resample(wavelength)
            self.wavelength[i] = wavelength
            self.flux[i] = spec.flux
        return self
