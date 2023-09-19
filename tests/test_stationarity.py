import numpy as np
import pandas as pd
import pytest

from eristropy.stationarity import StationarySignals


@pytest.fixture
def sample_df():
    signal_ids = np.repeat(["abc", "def"], 100)
    timestamps = np.tile(np.arange(100), 2)
    abc_values = np.linspace(0, 100, 100)
    def_values = np.sin(np.linspace(0, 2 * np.pi, 100))
    values = np.concatenate((abc_values, def_values))

    df = pd.DataFrame(
        {"signal_id": signal_ids, "timestamp": timestamps, "value": values}
    )
    return df


def test_invalid_method(sample_df):
    with pytest.raises(ValueError):
        _ = StationarySignals(sample_df, method="invalid")


def test_invalid_detrend_type(sample_df):
    with pytest.raises(ValueError):
        _ = StationarySignals(sample_df, detrend_type="invalid")


def test_invalid_alpha(sample_df):
    with pytest.raises(ValueError):
        _ = StationarySignals(sample_df, alpha=-0.1)


def test_invalid_ls_range(sample_df):
    with pytest.raises(ValueError):
        _ = StationarySignals(sample_df, ls_range=(0, 50))


def test_invalid_n_searches(sample_df):
    with pytest.raises(ValueError):
        _ = StationarySignals(sample_df, n_searches=-1)


def test_invalid_n_splits(sample_df):
    with pytest.raises(ValueError):
        _ = StationarySignals(sample_df, n_splits=0)


def test_invalid_eps(sample_df):
    with pytest.raises(ValueError):
        _ = StationarySignals(sample_df, eps=-0.1)


def test_invalid_gp_implementation(sample_df):
    with pytest.raises(ValueError):
        _ = StationarySignals(sample_df, gp_implementation="invalid")


def test_calculate_pvalues(sample_df):
    signals = StationarySignals(sample_df, method="difference")
    pvalues = signals._calculate_pvalues(sample_df)

    # Check that all p-values are between 0 and 1
    assert np.all((pvalues >= 0) & (pvalues <= 1))


def test_determine_stationary_signals(sample_df):
    signals = StationarySignals(sample_df, method="difference")
    signals._determine_stationary_signals(sample_df)

    assert signals.stationary_frac_ == 0.5
    assert np.array_equal(signals.stationary_signals_, np.array(["def"]))


def test_make_stationary_signals():
    signal_ids = np.repeat(["abc", "def"], 100)
    timestamps = np.tile(np.arange(100), 2)
    rng = np.random.default_rng(17)
    abc_values = rng.uniform(-5, 5, size=(100,))
    def_values = rng.uniform(-5, 5, size=(100,))
    values = np.concatenate((abc_values, def_values))
    df = pd.DataFrame(
        {"signal_id": signal_ids, "timestamp": timestamps, "value": values}
    )

    # Test case: Example in documentation
    signals = StationarySignals(df, method="difference", normalize_signals=False)
    result = signals.make_stationary_signals()

    abc_diff = np.diff(df.loc[df.signal_id == "abc", "value"].values)

    def_diff = np.diff(df.loc[df.signal_id == "def", "value"].values)
    values = np.concatenate((abc_diff, def_diff))

    expected_result = pd.DataFrame(
        {
            "signal_id": np.repeat(["abc", "def"], 99),
            "timestamp": np.tile(np.arange(1, 100), 2),
            "value": values,
        }
    )

    pd.testing.assert_frame_equal(result, expected_result)
