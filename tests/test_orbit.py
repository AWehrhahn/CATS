import numpy as np

import pytest
from cats.orbit import Orbit as orbit_calculator

@pytest.fixture
def orbit(configuration, parameters):
    oc = orbit_calculator(configuration, parameters)
    return oc

@pytest.fixture
def times(parameters):
    first = parameters["transit"].mjd - parameters["period"].to("day").value / 4
    last = parameters["transit"].mjd + parameters["period"].to("day").value / 4
    return np.linspace(first, last, 10)

def test_maxphase(orbit):
    maxphase = orbit.maximum_phase()
    assert isinstance(maxphase, (float, np.floating))
    assert maxphase >= 0 and maxphase <= 2 * np.pi 

def test_phases(orbit, times):
    phases = orbit.get_phase(times)

    assert isinstance(phases, np.ndarray)
    assert phases.ndim == 1
    assert phases.size == times.size
    assert np.all((phases >= 0) & (phases <= 2 * np.pi))

