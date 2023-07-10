import numpy as np

from cpyet._bootstrap import (
    _optimal_block_size,
    _get_idx,
    _single_stationary_boot,
    _stationary_bootstrap,
)

from cpyet.utils import _seed


def test_optimal_block_size():
    x = np.arange(100, dtype=np.float64)
    rng = np.random.default_rng(17)
    noise = rng.normal(scale=0.25, size=(x.size,))
    y = x + noise

    def _arch_optimal_block(x: np.ndarray) -> float:
        nobs = x.shape[0]
        eps = x - x.mean(0)
        b_max = np.ceil(min(3 * np.sqrt(nobs), nobs / 3))
        kn = max(5, int(np.log10(nobs)))
        m_max = int(np.ceil(np.sqrt(nobs))) + kn
        # Find first collection of kn autocorrelations that are insignificant
        cv = 2 * np.sqrt(np.log10(nobs) / nobs)
        acv = np.zeros(m_max + 1)
        abs_acorr = np.zeros(m_max + 1)
        opt_m: int | None = None
        for i in range(m_max + 1):
            v1 = eps[i + 1 :] @ eps[i + 1 :]
            v2 = eps[: -(i + 1)] @ eps[: -(i + 1)]
            cross_prod = eps[i:] @ eps[: nobs - i]
            acv[i] = cross_prod / nobs
            abs_acorr[i] = np.abs(cross_prod) / np.sqrt(v1 * v2)
            if i >= kn:
                if np.all(abs_acorr[i - kn : i] < cv) and opt_m is None:
                    opt_m = i - kn
        m = 2 * max(opt_m, 1) if opt_m is not None else m_max
        m = min(m, m_max)

        g = 0.0
        lr_acv = acv[0]
        for k in range(1, m + 1):
            lam = 1 if k / m <= 1 / 2 else 2 * (1 - k / m)
            g += 2 * lam * k * acv[k]
            lr_acv += 2 * lam * acv[k]
        d_sb = 2 * lr_acv**2
        b_sb = ((2 * g**2) / d_sb) ** (1 / 3) * nobs ** (1 / 3)
        b_sb = min(b_sb, b_max)
        return b_sb

    result = _optimal_block_size(y, c=2.0)
    expected_result = _arch_optimal_block(y)
    np.testing.assert_almost_equal(result, expected_result)


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
    n_boot = 10
    c = 2.0

    # Execute
    _seed(17)
    X = _stationary_bootstrap(x, n_boot, c)

    # Test 1: Check output shape
    assert X.shape == (n_boot, len(x)), "Output shape mismatch"

    # Test 2: Check output data type
    assert X.dtype == np.float64, "Data type of output is incorrect"

    # Test 3: Check elements in output are in the original time series
    for i in range(n_boot):
        assert np.all(
            np.isin(X[i, :], x)
        ), "Elements in bootstrapped sample not in original series"
