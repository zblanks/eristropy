import numpy as np
import pandas as pd
import pytest

from eristropy.stationarity import (
    _calculate_pvalues,
    determine_stationary_signals,
    make_stationary_signals,
)


def test_calculate_pvalues():
    # Define the example data
    signal_ids = np.repeat(["abc", "def"], 100)
    timestamps = np.tile(np.arange(100), 2)
    abc_values = np.linspace(0, 100, 100)
    def_values = np.sin(np.linspace(0, 2 * np.pi, 100))
    values = np.concatenate((abc_values, def_values))
    df = pd.DataFrame(
        {"signal_id": signal_ids, "timestamp": timestamps, "value": values}
    )

    # Call the function to calculate p-values
    pvalues = _calculate_pvalues(df)

    # Compare the computed p-values with the expected values
    expected_pvalues = np.array([0.9134984832798951, 0.0])

    np.testing.assert_allclose(pvalues, expected_pvalues)


def test_determine_stationary_signals():
    # Test case 1: When required columns are missing in the DataFrame
    df = pd.DataFrame({"signal_id": ["abc", "def"], "value": [1, 2]})
    with pytest.raises(ValueError):
        determine_stationary_signals(df)

    # Test case 2: When DataFrame is empty
    df = pd.DataFrame(columns=["signal_id", "timestamp", "value"])
    with pytest.raises(ValueError):
        determine_stationary_signals(df)

    # Test case 3: When significance level is invalid
    df = pd.DataFrame(
        {"signal_id": ["abc", "def", "abc"], "timestamp": [1, 2, 3], "value": [1, 2, 3]}
    )
    with pytest.raises(ValueError):
        determine_stationary_signals(df, alpha=0)
    with pytest.raises(ValueError):
        determine_stationary_signals(df, alpha=1)
    with pytest.raises(ValueError):
        determine_stationary_signals(df, alpha=-0.05)

    # Test case 4: When no unique signal IDs are found
    df = pd.DataFrame({"signal_id": [], "timestamp": [], "value": []})
    with pytest.raises(ValueError):
        determine_stationary_signals(df)

    # Test case 5: Example from documentation
    signal_ids = np.repeat(["abc", "def"], 100)
    timestamps = np.tile(np.arange(100), 2)
    abc_values = np.linspace(0, 100, 100)
    def_values = np.sin(np.linspace(0, 2 * np.pi, 100))
    values = np.concatenate((abc_values, def_values))
    df = pd.DataFrame(
        {"signal_id": signal_ids, "timestamp": timestamps, "value": values}
    )
    stationary_fraction, stationary_signals = determine_stationary_signals(df)
    assert stationary_fraction == 0.5
    assert np.array_equal(stationary_signals, np.array(["def"]))


@pytest.fixture
def sample_dataframe():
    signal_ids = np.repeat(["abc", "def"], 100)
    timestamps = np.tile(np.arange(100), 2)
    rng = np.random.default_rng(17)
    abc_values = rng.uniform(-5, 5, size=(100,))
    def_values = rng.uniform(-5, 5, size=(100,))
    values = np.concatenate((abc_values, def_values))
    df = pd.DataFrame(
        {"signal_id": signal_ids, "timestamp": timestamps, "value": values}
    )
    return df


def test_make_stationary_signals(sample_dataframe):
    with pytest.raises(TypeError, match="Input must be a pandas DataFrame."):
        make_stationary_signals(None, method="difference")

    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        make_stationary_signals(pd.DataFrame(), method="difference")

    with pytest.raises(ValueError, match="Missing required columns: {'signal_id'}"):
        make_stationary_signals(
            sample_dataframe.drop(columns="signal_id"), method="difference"
        )

    # Create a copy of the sample_dataframe fixture
    modified_dataframe = sample_dataframe.copy()

    with pytest.raises(
        ValueError,
        match="Input DataFrame contains NaN or np.inf values or non-numeric data.",
    ):
        modified_dataframe.loc[0, "value"] = np.inf
        make_stationary_signals(modified_dataframe, method="difference")

    with pytest.raises(
        ValueError,
        match="Input DataFrame contains NaN or np.inf values or non-numeric data.",
    ):
        modified_dataframe.loc[0, "value"] = "not a number"
        make_stationary_signals(modified_dataframe, method="difference")

    with pytest.raises(
        ValueError,
        match="Invalid value for 'method'. Must be either 'difference' or 'detrend'.",
    ):
        make_stationary_signals(sample_dataframe, method="invalid")

    with pytest.raises(
        ValueError,
        match="Invalid value for 'detrend_type'. Must be either 'lr' or 'gp'.",
    ):
        make_stationary_signals(
            sample_dataframe, method="detrend", detrend_type="invalid"
        )

    with pytest.raises(ValueError) as exc_info:
        make_stationary_signals(sample_dataframe, method="difference", alpha=0)
    assert "Significance level must be between 0 and 1 (exclusive)" in str(
        exc_info.value
    )

    with pytest.raises(
        ValueError, match="The lower bound of 'ls_range' must be greater than 0."
    ):
        make_stationary_signals(
            sample_dataframe, method="detrend", detrend_type="gp", ls_range=(0, 100)
        )

    with pytest.raises(ValueError) as exc_info:
        make_stationary_signals(
            sample_dataframe, method="detrend", detrend_type="gp", n_searches=-10
        )
    assert "The number of searches (n_searches) must be a positive integer." in str(
        exc_info.value
    )

    with pytest.raises(ValueError) as exc_info:
        make_stationary_signals(
            sample_dataframe, method="detrend", detrend_type="gp", n_splits=-5
        )
    assert "The number of splits (n_splits) must be a positive integer." in str(
        exc_info.value
    )

    with pytest.raises(
        ValueError, match="ls_range must be a tuple of two np.number values."
    ):
        make_stationary_signals(
            sample_dataframe, method="detrend", detrend_type="gp", ls_range=(1.0, "100")
        )

    with pytest.raises(ValueError) as exc_info:
        make_stationary_signals(
            sample_dataframe, method="detrend", detrend_type="gp", random_seed="seed"
        )
    assert "random_seed must be None or an integer." in str(exc_info.value)

    # Test case: `gp_implementation` argument
    with pytest.raises(ValueError) as exc_info:
        make_stationary_signals(
            sample_dataframe,
            method="detrend",
            detrend_type="gp",
            gp_implementation="meow",
        )
        assert (
            "Invalid value for 'gp_implementation'. Must be either 'sklearn' or 'numba'."
            in str(exc_info.value)
        )

    # Test case: Example in documentation
    result = make_stationary_signals(sample_dataframe, method="difference")

    abc_diff = np.diff(
        sample_dataframe.loc[sample_dataframe.signal_id == "abc", "value"].values
    )

    def_diff = np.diff(
        sample_dataframe.loc[sample_dataframe.signal_id == "def", "value"].values
    )
    values = np.concatenate((abc_diff, def_diff))

    expected_result = pd.DataFrame(
        {
            "signal_id": np.repeat(["abc", "def"], 99),
            "timestamp": np.tile(np.arange(1, 100), 2),
            "value": values,
        }
    )

    pd.testing.assert_frame_equal(result, expected_result)
