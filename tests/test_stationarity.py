import numpy as np
import pandas as pd
import pytest

from cpyet.stationarity import (
    _difference,
    _difference_all_signals,
    _detrend_linreg,
    _detrend_all_signals_linreg,
    _squared_euclidean_distance_xx,
    _squared_euclidean_distance_xy,
    _rbf_kernel,
    _time_series_split,
    _mean_squared_error,
    _mean_error_over_splits,
    _find_best_ls,
    _detrend_gp,
    _detrend_all_signals_gp,
    _calculate_pvalues,
    determine_stationary_signals,
    make_stationary_signals,
)


def test_difference():
    # Test case 1: Valid input
    x = [1, 3, 6, 10, 15]
    expected_result = np.array([2, 3, 4, 5])
    assert np.array_equal(_difference(x), expected_result)

    # Test case 2: Input with NaN value
    x_nan = [1, 2, np.nan, 4, 5]
    with pytest.raises(ValueError):
        _difference(x_nan)

    # Test case 3: Input with Inf value
    x_inf = [1, 2, np.inf, 4, 5]
    with pytest.raises(ValueError):
        _difference(x_inf)

    # Test case 4: Input of insufficient length -- less than 2 elements
    x_short = [1]
    with pytest.raises(ValueError):
        _difference(x_short)


def test_difference_all_signals():
    # Test case 1: Example provided in function documentation
    df = pd.DataFrame(
        {
            "signal_id": ["abc", "abc", "def", "def"],
            "timestamp": [1, 2, 1, 2],
            "value": [2, 3, 5, 7],
        }
    )

    expected_result = pd.DataFrame(
        {"signal_id": ["abc", "def"], "timestamp": [2, 2], "value": [1, 2]}
    )

    result = _difference_all_signals(df)
    pd.testing.assert_frame_equal(result, expected_result)


def test_detrend_linreg():
    # Test case 1: Example from the documentation
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 3, 5, 7, 8])
    expected_result = np.array([0.2, -0.4, 0.0, 0.4, -0.2])

    result = _detrend_linreg(x, y)
    np.testing.assert_allclose(result, expected_result)


def test_detrend_all_signals_linreg():
    # Test case 1: Example in documentation
    df = pd.DataFrame(
        {
            "signal_id": ["abc", "abc", "def", "def"],
            "timestamp": [1, 2, 1, 2],
            "value": [2, 3, 5, 7],
        }
    )

    # Expected result
    expected_result = pd.DataFrame(
        {
            "signal_id": ["abc", "abc", "def", "def"],
            "timestamp": [1, 2, 1, 2],
            "value": [0.0, 0.0, 0.0, 0.0],
        }
    )

    # Compute the detrended signals
    result = _detrend_all_signals_linreg(df)

    # Assert that the result matches the expected result
    pd.testing.assert_frame_equal(result, expected_result)


def test_squared_euclidean_distance_xx():
    # Test case 1: Example in the docstring
    x = np.array([[1, 2], [3, 4], [5, 6]])

    distances_xx = _squared_euclidean_distance_xx(x)
    expected_xx = np.array([[0, 8, 32], [8, 0, 8], [32, 8, 0]])
    assert np.array_equal(distances_xx, expected_xx)


def test_squared_euclidean_distance_xy():
    x = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([[2, 2], [4, 4]])

    distances_xy = _squared_euclidean_distance_xy(x, y)
    expected_xy = np.array([[1, 13], [5, 1], [25, 5]])
    assert np.array_equal(distances_xy, expected_xy)


def test_rbf_kernel():
    # Test case 1: Example in documentation
    x = np.array([[1, 2], [3, 4], [5, 6]])
    D = _squared_euclidean_distance_xx(x)
    ls = 0.5

    expected_result = np.array(
        [
            [1.00000000e00, 1.12535175e-07, 1.60381089e-28],
            [1.12535175e-07, 1.00000000e00, 1.12535175e-07],
            [1.60381089e-28, 1.12535175e-07, 1.00000000e00],
        ]
    )

    result = _rbf_kernel(D, ls)
    np.testing.assert_allclose(result, expected_result)


def test_time_series_split():
    # Test case 1: Example in documentation
    x = np.arange(6)
    n_splits = 3

    splits = _time_series_split(x, n_splits)
    expected_splits = [([0, 1, 2], [3]), ([0, 1, 2, 3], [4]), ([0, 1, 2, 3, 4], [5])]

    for i, (train_indices, test_indices) in enumerate(splits):
        expected_train_indices, expected_test_indices = expected_splits[i]
        assert np.array_equal(train_indices, expected_train_indices)
        assert np.array_equal(test_indices, expected_test_indices)


def test_mean_squared_error():
    # Test case 1: Example in documentation
    y = np.array([1, 2, 3])
    yhat = np.array([1.5, 2.2, 2.8])
    expected_mse = 0.11

    mse = _mean_squared_error(y, yhat)

    assert np.isclose(
        mse, expected_mse
    ), f"Expected MSE: {expected_mse}, Actual MSE: {mse}"


def test_mean_error_over_splits():
    # Test case 1: Example in documentation
    X = np.arange(10).reshape(-1, 1)
    rng = np.random.default_rng(17)
    y = rng.normal(size=(X.shape[0],))
    ls = 0.5
    mean_error = _mean_error_over_splits(X, y, ls, n_splits=3)
    expected_error = 0.7484052691169865

    np.testing.assert_almost_equal(mean_error, expected_error)


def test_find_best_ls():
    # Test case 1: Example in docstring
    X = np.arange(10).reshape(-1, 1)
    rng = np.random.default_rng(17)
    y = rng.normal(size=(X.shape[0],))
    ls_vals = np.array([0.5, 1.0])
    best_ls = _find_best_ls(X, y, ls_vals, n_splits=3)
    expected_best_ls = 0.5

    np.testing.assert_almost_equal(best_ls, expected_best_ls)


def test_detrend_gp():
    # Test case 1: Example in documentation
    X = np.arange(10).reshape(-1, 1)
    rng = np.random.default_rng(17)
    y = rng.normal(size=(X.shape[0],))
    ls_vals = np.array([0.5, 1.0])
    n_splits = 3
    eps = 1e-6

    detrended_signal = _detrend_gp(X, y, ls_vals, n_splits, eps)
    expected_detrended_signal = np.array(
        [
            1.06695763e-06,
            2.54575510e-07,
            -4.44978049e-07,
            -9.54630765e-07,
            -1.81473399e-06,
            3.67333456e-07,
            -7.57561006e-07,
            -7.54210002e-07,
            -1.14763661e-07,
            -3.60613987e-08,
        ]
    )

    np.testing.assert_allclose(detrended_signal, expected_detrended_signal, atol=eps)


def test_detrend_all_signals_gp():
    # Test case 1: Example in docstring
    df = pd.DataFrame(
        {
            "signal_id": [
                "abc",
                "abc",
                "abc",
                "abc",
                "abc",
                "def",
                "def",
                "def",
                "def",
                "def",
            ],
            "timestamp": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "value": [10, 12, 15, 17, 18, 5, 6, 7, 11, 14],
        }
    )
    ls_vals = np.array([0.5, 1.0])

    detrended_df = _detrend_all_signals_gp(df, ls_vals, n_splits=3)

    # Assert the structure and values of the detrended_df
    expected_df = pd.DataFrame(
        {
            "signal_id": [
                "abc",
                "abc",
                "abc",
                "abc",
                "abc",
                "def",
                "def",
                "def",
                "def",
                "def",
            ],
            "timestamp": [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "value": [
                9.138937e-06,
                -1.467807e-06,
                1.298148e-05,
                -1.006977e-06,
                1.686713e-05,
                3.581833e-06,
                1.587954e-06,
                3.257620e-06,
                8.809471e-07,
                1.300595e-05,
            ],
        }
    )

    pd.testing.assert_frame_equal(detrended_df, expected_df)


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


def test_make_stationary_signals_exceptions(sample_dataframe):
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
