import numpy as np
from scipy.stats import ks_2samp

from eristropy.utils import (
    _mean,
    _standard_error,
    _unif_to_geom,
    _mean_squared_error,
    _squared_euclidean_distance_xx,
    _squared_euclidean_distance_xy,
)


def test_mean():
    # Make sure the output is the same as np.mean()
    x = np.arange(10, dtype=np.float64)
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
