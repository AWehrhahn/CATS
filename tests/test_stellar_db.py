import pytest

from exoorbit.bodies import Star
from cats.data_modules.stellar_db import StellarDb


# @pytest.fixture
# def parameters_expected_outputs():
#     return [("period", "day"), ("periastron", "jd"), ("transit", "jd"), ("teff", "K"), ("logg", ""), ("monh", ""), ("r_star", "km"),
#             ("m_star", "kg"), ("r_planet", "km"), ("m_planet", "kg"), ("sma", "km"), ("inc", "deg"), ("ecc", ""), ("w", "deg")]


def test_load_data(star):
    sdb = StellarDb()
    star_dict = sdb.get(star)

    assert isinstance(star_dict, Star)
