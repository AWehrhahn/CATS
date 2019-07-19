import numpy as np
import pytest

from cats import solution

def test_tikhonov():
    # f * x = g
    f = np.ones(10)
    g = np.ones(10) * 2
    l = 0
    t = solution.Tikhonov(f, g, l)
    assert isinstance(t, np.ndarray)
    assert t.ndim == 1
    assert t.size == 10
    assert np.all(t == 2)

def test_inputs_tikhonov():
    with pytest.raises(TypeError):
        solution.Tikhonov(None, None, None)
