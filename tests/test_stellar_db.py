import pytest
from astropy.units import UnitConversionError
from astropy.time import Time

from cats.data_modules.stellar_db import stellar_db



def test_load_data(star, planet, configuration, parameters_expected_outputs):
    sdb = stellar_db(configuration)
    par = sdb.get_parameters()

    assert isinstance(par, dict)
    for quantity, unit in parameters_expected_outputs:
        assert quantity in par.keys()
        try:
            if unit != "jd":
                par[quantity].to(unit)
                assert True
            else:
                assert isinstance(par[quantity], Time)
        except UnitConversionError:
            assert False, f"{quantity} is in an incompatible units. Expected {unit}, but got {par[quantity].unit}"
