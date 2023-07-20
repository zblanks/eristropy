import numpy as np
import pandas as pd

from eristropy.dataclasses import StationarySignalParams
from eristropy.linreg import _detrend_linreg, _detrend_all_signals_linreg


def test_detrend_linreg():
    # Test case 1: Example from the documentation
    X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    y = np.array([2, 3, 5, 7, 8])

    # Known, linear algebra solution to linear least squares
    Xint = np.column_stack((X, np.ones((X.shape[0], 1))))
    beta = np.linalg.lstsq(Xint, y, rcond=None)[0]
    yhat = Xint @ beta
    expected_result = y - yhat
    # expected_result = np.array([0.2, -0.4, 0.0, 0.4, -0.2])

    result = _detrend_linreg(X, y)
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
    params = StationarySignalParams(method="detrend", detrend_type="lr")
    result = _detrend_all_signals_linreg(df, params)

    # Assert that the result matches the expected result
    pd.testing.assert_frame_equal(result, expected_result)
