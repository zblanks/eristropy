import numpy as np
import pandas as pd

from eristropy.dataclasses import StationarySignalParams
from eristropy.stationarity import StationarySignals


def test_calculate_pvalues():
    signal_ids = np.repeat(["abc", "def"], 100)
    timestamps = np.tile(np.arange(100), 2)
    abc_values = np.linspace(0, 100, 100)
    def_values = np.sin(np.linspace(0, 2 * np.pi, 100))
    values = np.concatenate((abc_values, def_values))
    df = pd.DataFrame(
        {"signal_id": signal_ids, "timestamp": timestamps, "value": values}
    )

    params = StationarySignalParams(method="difference")
    signals = StationarySignals(df, params)
    pvalues = signals._calculate_pvalues()

    # Compare the computed p-values with the expected values
    expected_pvalues = np.array([0.9134984832798951, 0.0])

    np.testing.assert_allclose(pvalues, expected_pvalues)


def test_determine_stationary_signals():
    signal_ids = np.repeat(["abc", "def"], 100)
    timestamps = np.tile(np.arange(100), 2)
    abc_values = np.linspace(0, 100, 100)
    def_values = np.sin(np.linspace(0, 2 * np.pi, 100))
    values = np.concatenate((abc_values, def_values))
    df = pd.DataFrame(
        {"signal_id": signal_ids, "timestamp": timestamps, "value": values}
    )

    params = StationarySignalParams(method="difference")
    signals = StationarySignals(df, params)
    signals._determine_stationary_signals()

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
    params = StationarySignalParams(method="difference")
    signals = StationarySignals(df, params)
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
