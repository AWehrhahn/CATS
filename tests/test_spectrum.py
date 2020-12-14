import pytest
import tempfile
import numpy as np
from astropy import units
import os

from cats.spectrum import Spectrum1D, SpectrumArray


@pytest.fixture(params=["flex", "fits"])
def formats(request):
    return request.param


@pytest.fixture(scope="function")
def filename(formats):
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{formats}")
    yield tmp_file.name
    try:
        os.delete(tmp_file.name)
    except:
        pass


@pytest.fixture
def wavelength():
    wave = np.linspace(1000, 2000, 100)
    wave = wave << units.Unit("AA")
    return wave


@pytest.fixture
def flux():
    flux = np.full(100, 2)
    flux = flux << units.one
    return flux


@pytest.fixture
def spec1d(wavelength, flux):
    spec = Spectrum1D(spectral_axis=wavelength, flux=flux)
    return spec


def test_create_spectrum_1d(wavelength, flux):
    spec = Spectrum1D(spectral_axis=wavelength, flux=flux)
    assert isinstance(spec, Spectrum1D)
    assert np.all(spec.wavelength == wavelength)
    assert np.all(spec.flux == flux)


def test_read_write(spec1d, filename, formats):
    spec1d.write(filename, format=formats)

    # read with format specified
    spec2 = spec1d.read(filename, format=formats)
    # read without format specified
    spec2 = spec1d.read(filename)

    assert isinstance(spec2, spec1d.__class__)
    assert np.all(spec2.wavelength == spec1d.wavelength)
    assert np.all(spec2.flux == spec1d.flux)
