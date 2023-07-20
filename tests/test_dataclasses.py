import pytest
from eristropy.dataclasses import StationarySignalParams


def test_invalid_method():
    with pytest.raises(ValueError):
        _ = StationarySignalParams(method="invalid")


def test_invalid_detrend_type():
    with pytest.raises(ValueError):
        _ = StationarySignalParams(detrend_type="invalid")


def test_invalid_alpha():
    with pytest.raises(ValueError):
        _ = StationarySignalParams(alpha=-0.1)


def test_invalid_ls_range():
    with pytest.raises(ValueError):
        _ = StationarySignalParams(ls_range=(0, 50))


def test_invalid_n_searches():
    with pytest.raises(ValueError):
        _ = StationarySignalParams(n_searches=-1)


def test_invalid_n_splits():
    with pytest.raises(ValueError):
        _ = StationarySignalParams(n_splits=0)


def test_invalid_eps():
    with pytest.raises(ValueError):
        _ = StationarySignalParams(eps=-0.1)


def test_invalid_gp_implementation():
    with pytest.raises(ValueError):
        _ = StationarySignalParams(gp_implementation="invalid")
