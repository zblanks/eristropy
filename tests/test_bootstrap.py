import numpy as np

from eristropy._bootstrap import (
    _get_idx,
    _single_stationary_boot,
    _stationary_bootstrap,
)

from eristropy.utils import _seed


def test_get_idx():
    n = 100
    p = 0.5
    idx = _get_idx(n, p)

    # Test 1: Check return type
    assert isinstance(idx, np.ndarray), "Expected return type to be numpy array"
    assert idx.dtype == np.int32, "Expected dtype to be int32"

    # Test 2: Check array length
    assert len(idx) == min(
        idx[-1] - idx[0] + 1, n
    ), "Unexpected length of returned array"

    # Test 3: Check content of array
    assert np.all(idx < n), "Expected all indices to be less than n"


def test_single_stationary_boot():
    n = 100
    p = 0.5
    arr = _single_stationary_boot(n, p)

    # Test 1: Check return type
    assert isinstance(arr, np.ndarray), "Expected return type to be numpy array"
    assert arr.dtype == np.int32, "Expected dtype to be int32"

    # Test 2: Check array length
    assert len(arr) == n, "Unexpected length of returned array"

    # Test 3: Check content of array
    assert np.all(arr >= 0), "Expected all indices to be non-negative"
    assert np.all(arr < n), "Expected all indices to be less than n"

    # Test 4: Check randomness
    arr1 = _single_stationary_boot(n, p)
    arr2 = _single_stationary_boot(n, p)
    assert not np.array_equal(
        arr1, arr2
    ), "Expected different arrays over multiple calls"


def test_stationary_bootstrap():
    # Set up for reproducibility
    rng = np.random.default_rng(17)
    x = rng.random(100)
    p = 0.5
    n_boot = 10

    # Execute
    _seed(17)
    X = _stationary_bootstrap(x, p, n_boot)

    # Test 1: Check output shape
    assert X.shape == (n_boot, len(x)), "Output shape mismatch"

    # Test 2: Check output data type
    assert X.dtype == np.float64, "Data type of output is incorrect"

    # Test 3: Check elements in output are in the original time series
    for i in range(n_boot):
        assert np.all(
            np.isin(X[i, :], x)
        ), "Elements in bootstrapped sample not in original series"
