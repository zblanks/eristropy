import numpy as np
import pandas as pd
import pytest

from eristropy.difference import _difference, _difference_all_signals


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

    result = _difference_all_signals(df, "signal_id", "timestamp", "value")
    pd.testing.assert_frame_equal(result, expected_result)
