import numpy as np
import pytest

from cpyet.stationarity import difference


def test_difference():
    # Test case 1: Valid input
    x = [1, 3, 6, 10, 15]
    expected_result = np.array([2, 3, 4, 5])
    assert np.array_equal(difference(x), expected_result)

    # Test case 2: Invalid input type
    invalid_input = "not an array"
    with pytest.raises(TypeError):
        difference(invalid_input)

    # Test case 3: Input with NaN value
    x_nan = [1, 2, np.nan, 4, 5]
    with pytest.raises(ValueError):
        difference(x_nan)

    # Test case 4: Input with Inf value
    x_inf = [1, 2, np.inf, 4, 5]
    with pytest.raises(ValueError):
        difference(x_inf)

    # Test case 5: Input of insufficient length -- less than 2 elements
    x_short = [1]
    with pytest.raises(ValueError):
        difference(x_short)
