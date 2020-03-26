import pytest

# TODO
# Tests for everything


@pytest.fixture(params=[("GJ1214", "b")], ids=["GJ1214_b"])
def dataset(request):
    return request.param


@pytest.fixture
def star(dataset):
    return dataset[0]


@pytest.fixture
def planet(dataset):
    return dataset[1]
