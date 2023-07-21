import pytest
from eristropy.dataclasses import (
    SampEnSettings,
    SampEnSettingWarning,
    StationarySignalParams,
)


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


def test_sampensettings_check_ranges():
    # Invalid r_range
    with pytest.raises(ValueError):
        SampEnSettings(r_range=(0.50, 0.10))  # second value less than first
    with pytest.raises(ValueError):
        SampEnSettings(r_range=(-0.10, 0.50))  # first value less than 0

    # Invalid m_range
    with pytest.raises(ValueError):
        SampEnSettings(m_range=(3, 1))  # second value less than first
    with pytest.raises(ValueError):
        SampEnSettings(m_range=(1.5, 3))  # first value not an integer
    with pytest.raises(ValueError):
        SampEnSettings(m_range=(1, 3.5))  # second value not an integer

    # Invalid p_range
    with pytest.raises(ValueError):
        SampEnSettings(p_range=(0.99, 0.01))  # second value less than first
    with pytest.raises(ValueError):
        SampEnSettings(p_range=(-0.01, 0.99))  # first value less than 0
    with pytest.raises(ValueError):
        SampEnSettings(p_range=(0.01, 1.1))  # second value greater than 1


def test_sampensettings_check_fixed_values():
    # Invalid m
    with pytest.raises(ValueError):
        SampEnSettings(m=-1)  # m less than 0
    with pytest.raises(ValueError):
        SampEnSettings(m=0)  # m equals 0
    with pytest.raises(ValueError):
        SampEnSettings(m=1.5)  # m not an integer

    # Invalid r
    with pytest.raises(ValueError):
        SampEnSettings(r=-0.1)  # r less than 0
    with pytest.raises(ValueError):
        SampEnSettings(r=0)  # r equals 0

    # Invalid p
    with pytest.raises(ValueError):
        SampEnSettings(p=-0.1)  # p less than 0
    with pytest.raises(ValueError):
        SampEnSettings(p=1.1)  # p greater than 1
    with pytest.raises(ValueError):
        SampEnSettings(p=1)  # p equals 1
    with pytest.raises(ValueError):
        SampEnSettings(p=0)  # p equals 0

    # Invalid lam1
    with pytest.raises(ValueError):
        SampEnSettings(lam1=-0.1)  # lam1 less than 0
    with pytest.raises(ValueError):
        SampEnSettings(lam1=0)  # lam1 equals 0


def test_sampensettings_check_positive_integer():
    # Invalid n_boot
    with pytest.raises(ValueError):
        SampEnSettings(n_boot=-1)  # n_boot less than 0
    with pytest.raises(ValueError):
        SampEnSettings(n_boot=0)  # n_boot equals 0
    with pytest.raises(ValueError):
        SampEnSettings(n_boot=1.5)  # n_boot not an integer

    # Invalid n_trials
    with pytest.raises(ValueError):
        SampEnSettings(n_trials=-1)  # n_trials less than 0
    with pytest.raises(ValueError):
        SampEnSettings(n_trials=0)  # n_trials equals 0
    with pytest.raises(ValueError):
        SampEnSettings(n_trials=1.5)  # n_trials not an integer

    # Invalid random_seed
    with pytest.raises(ValueError):
        SampEnSettings(random_seed=-1)  # random_seed less than 0
    with pytest.raises(ValueError):
        SampEnSettings(random_seed=1.5)  # random_seed not an integer


def test_sampensettings_check_string_attrs():
    # Invalid signal_id
    with pytest.raises(ValueError):
        SampEnSettings(signal_id=123)  # signal_id is not a string

    # Invalid timestamp
    with pytest.raises(ValueError):
        SampEnSettings(timestamp=123)  # timestamp is not a string

    # Invalid value_col
    with pytest.raises(ValueError):
        SampEnSettings(value_col=123)  # value_col is not a string


def test_sampensettings_default_values():
    sampen_settings = SampEnSettings()

    assert sampen_settings.signal_id == "signal_id"
    assert sampen_settings.timestamp == "timestamp"
    assert sampen_settings.value_col == "value"
    assert sampen_settings.n_boot == 100
    assert sampen_settings.n_trials == 100
    assert sampen_settings.random_seed is None
    assert sampen_settings.r_range == (0.10, 0.50)
    assert sampen_settings.m_range == (1, 3)
    assert sampen_settings.p_range == (0.01, 0.99)
    assert sampen_settings.lam1 == 0.33
    assert sampen_settings.r is None
    assert sampen_settings.m is None
    assert sampen_settings.p is None


def test_sampensettings_custom_values():
    sampen_settings = SampEnSettings(
        signal_id="my_signal_id",
        timestamp="my_timestamp",
        value_col="my_value",
        n_boot=200,
        n_trials=300,
        random_seed=42,
        r_range=(0.15, 0.45),
        m_range=(2, 4),
        p_range=(0.05, 0.95),
        lam1=0.25,
        r=0.2,
        m=2,
        p=0.1,
    )

    assert sampen_settings.signal_id == "my_signal_id"
    assert sampen_settings.timestamp == "my_timestamp"
    assert sampen_settings.value_col == "my_value"
    assert sampen_settings.n_boot == 200
    assert sampen_settings.n_trials == 300
    assert sampen_settings.random_seed == 42
    assert sampen_settings.r_range == (0.15, 0.45)
    assert sampen_settings.m_range == (2, 4)
    assert sampen_settings.p_range == (0.05, 0.95)
    assert sampen_settings.lam1 == 0.25
    assert sampen_settings.r == 0.2
    assert sampen_settings.m == 2
    assert sampen_settings.p == 0.1


def test_sampensettings_warns_on_low_values():
    # Low n_boot
    with pytest.warns(SampEnSettingWarning):
        SampEnSettings(n_boot=1)

    # Low n_trials
    with pytest.warns(SampEnSettingWarning):
        SampEnSettings(n_trials=1)


def test_sampensettings_warns_on_extreme_fixed_values():
    # Extreme r
    with pytest.warns(SampEnSettingWarning):
        SampEnSettings(r=0.50)

    # Extreme p
    with pytest.warns(SampEnSettingWarning):
        SampEnSettings(p=0.01)

    test_sampensettings_warns_on_low_values()
