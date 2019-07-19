import pytest

from cats import config
from cats.data_modules.stellar_db import stellar_db

#TODO
# Tests for each data module
# Tests for orbit calculation
# Tests for solution (Tikhonov)
# Tests for best lambda finding

@pytest.fixture(params=[("GJ1214", "b")], ids=["GJ1214_b"])
def dataset(request):
    return request.param

@pytest.fixture
def star(dataset):
    return dataset[0]

@pytest.fixture
def planet(dataset):
    return dataset[1]

@pytest.fixture
def configuration(star, planet):
    return config.load_config(star, planet)

@pytest.fixture
def parameters_expected_outputs():
    return [("period", "day"), ("periastron", "jd"), ("transit", "jd"), ("teff", "K"), ("logg", ""), ("monh", ""), ("r_star", "km"),
            ("m_star", "kg"), ("r_planet", "km"), ("m_planet", "kg"), ("sma", "km"), ("inc", "deg"), ("ecc", ""), ("w", "deg")]

@pytest.fixture
def parameters(configuration):
    sdb = stellar_db(configuration)
    return sdb.get_parameters()
