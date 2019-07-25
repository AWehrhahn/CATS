import numpy as np

import pytest
from cats.orbit import Orbit as orbit_calculator

# Use Earth as test system?

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
    assert isinstance(maxphase, np.ndarray)
    assert maxphase.ndim == 1
    assert maxphase.size == 2
    assert np.all(maxphase >= -np.pi)
    assert np.all(maxphase <= np.pi)
    assert maxphase[0] < maxphase[1]
