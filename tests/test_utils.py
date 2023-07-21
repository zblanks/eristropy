import numpy as np
from scipy.stats import ks_2samp

from eristropy.utils import (
    _mean,
    _standard_error,
    _unif_to_geom,
    _mean_squared_error,
    _squared_euclidean_distance_xx,
    _squared_euclidean_distance_xy,
    _sampen,
)


def test_mean():
    # Make sure the output is the same as np.mean()
    x = np.arange(10, dtype=np.float64)
    result = _mean(x)
    expected_result = np.mean(x)
    np.testing.assert_almost_equal(result, expected_result)


def test_mean_different_sizes():
    # Check the function with a very large array
    x = np.random.rand(10**6)
    result = _mean(x)
    expected_result = np.mean(x)
    np.testing.assert_almost_equal(result, expected_result)


def test_standard_error():
    x = np.arange(100, dtype=np.float64)
    rng = np.random.default_rng(17)
    noise = rng.normal(scale=0.25, size=(x.size,))
    y = x + noise

    result = _standard_error(y)
    expected_result = np.sqrt(np.var(y))
    np.testing.assert_almost_equal(result, expected_result)


def test_standard_error_different_sizes():
    # Check the function with a very large array
    x = np.random.rand(10**6)
    result = _standard_error(x)
    expected_result = np.sqrt(np.var(x))
    np.testing.assert_almost_equal(result, expected_result)


def test_standard_error_different_data():
    # Check the function with a constant array
    x = np.full(100, 5, dtype=np.float64)
    result = _standard_error(x)
    expected_result = np.sqrt(np.var(x))
    np.testing.assert_almost_equal(result, expected_result)

    # Check the function with an ascending array
    x = np.arange(100, dtype=np.float64)
    result = _standard_error(x)
    expected_result = np.sqrt(np.var(x))
    np.testing.assert_almost_equal(result, expected_result)


def test_unif_to_geom():
    np.random.seed(17)  # For reproducibility
    n_samples = 100000
    u_samples = np.random.uniform(size=n_samples)

    p = 0.3  # Example probability
    transformed_samples = np.array([_unif_to_geom(u, p) for u in u_samples])

    geom_samples = np.random.geometric(p, size=n_samples)

    # Use Kolmogorov-Smirnov test to compare the transformed vs empirical
    # Geometric distribution samples
    pvalue = ks_2samp(transformed_samples, geom_samples)[1]
    assert pvalue > 0.05


def test_mean_squared_error():
    # Test case 1: Example in documentation
    y = np.array([1, 2, 3]).astype(np.float64)
    yhat = np.array([1.5, 2.2, 2.8])
    expected_mse = 0.11

    mse = _mean_squared_error(y, yhat)

    assert np.isclose(
        mse, expected_mse
    ), f"Expected MSE: {expected_mse}, Actual MSE: {mse}"


def test_mean_squared_error_different_sizes():
    # Check the function with a very large array
    y = np.random.rand(10**6)
    yhat = np.random.rand(10**6)
    mse = _mean_squared_error(y, yhat)
    expected_mse = ((y - yhat) ** 2).mean()
    np.testing.assert_almost_equal(mse, expected_mse)


def test_mean_squared_error_different_data():
    # Check the function with a constant array and correct predictions
    y = np.full(100, 5, dtype=np.float64)
    yhat = np.full(100, 5, dtype=np.float64)
    mse = _mean_squared_error(y, yhat)
    expected_mse = 0
    np.testing.assert_almost_equal(mse, expected_mse)

    # Check the function with an ascending array and constant error
    y = np.arange(100, dtype=np.float64)
    yhat = y + 2
    mse = _mean_squared_error(y, yhat)
    expected_mse = 4
    np.testing.assert_almost_equal(mse, expected_mse)

    # Check the function with an array with repeated elements and varying error
    y = np.array([1, 1, 2, 2, 3, 3], dtype=np.float64)
    yhat = np.array([1, 2, 2, 3, 3, 4], dtype=np.float64)
    mse = _mean_squared_error(y, yhat)
    expected_mse = ((y - yhat) ** 2).mean()
    np.testing.assert_almost_equal(mse, expected_mse)


def test_squared_euclidean_distance_xx():
    # Test case 1: Example in the docstring
    X = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float64)

    distances_xx = _squared_euclidean_distance_xx(X)
    expected_xx = np.array([[0, 8, 32], [8, 0, 8], [32, 8, 0]])
    assert np.array_equal(distances_xx, expected_xx)


def test_squared_euclidean_distance_xy():
    X = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float64)
    Y = np.array([[2, 2], [4, 4]]).astype(np.float64)

    distances_xy = _squared_euclidean_distance_xy(X, Y)
    expected_xy = np.array([[1, 13], [5, 1], [25, 5]])
    assert np.array_equal(distances_xy, expected_xy)


def test_sampen_constant_signal():
    # Constant signal should have zero SampEn
    x = np.full(1000, 5.0)
    m = 2
    r = 0.2
    sampen = _sampen(x, m, r)
    np.testing.assert_almost_equal(sampen, 0.0)


def test_sampen_linear_signal():
    # Linear signal should also have zero SampEn
    x = np.linspace(0, 1, 1000)
    m = 2
    r = 0.2
    sampen = _sampen(x, m, r)
    np.testing.assert_almost_equal(sampen, 0.0)
    assert sampen == 0.0


def test_sampen_random_signal():
    # Random signal should have relatively high SampEn
    rng = np.random.default_rng(17)
    x = rng.normal(size=1000)
    m = 2
    r = 0.2
    sampen = _sampen(x, m, r)
    assert sampen > 0.0


def test_sampen_repeat_pattern_signal():
    # Repeating pattern should have lower SampEn than random signal
    rng = np.random.default_rng(17)
    x = np.sin(np.linspace(0, 2 * np.pi, 1000))
    m = 2
    r = 0.2
    sampen = _sampen(x, m, r)
    assert sampen > 0.0 and sampen < _sampen(rng.normal(size=1000), m, r)


def test_sampen_increasing_r():
    # Increasing r should decrease SampEn
    rng = np.random.default_rng(17)
    x = rng.normal(size=1000)
    m = 2
    r1 = 0.2
    r2 = 0.3
    sampen1 = _sampen(x, m, r1)
    sampen2 = _sampen(x, m, r2)
    assert sampen2 < sampen1
