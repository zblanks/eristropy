import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel

from cpyet._gp import (
    _rbf_kernel,
    _time_series_split,
    _solve_cholesky,
    _jitter_kernel,
    _fit,
    _predict,
    _mean_error_over_splits,
    _find_best_ls,
    _detrend_gp,
    _detrend_all_signals_gp_numba,
)

from cpyet.utils import _squared_euclidean_distance_xx


def test_rbf_kernel():
    # Test case 1: Example in documentation
    X = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float64)
    D = _squared_euclidean_distance_xx(X)
    ls = 0.5

    # Known, valid scikit-learn implementation
    gamma = 0.5 * (1.0 / ls**2)
    expected_result = rbf_kernel(X, gamma=gamma)

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


def test_solve_cholesky():
    # Test case 1: Example in documentation
    K = np.array([[1, 1 / 2, 0], [1 / 2, 1, 1 / 3], [0, 1 / 3, 1]])
    y = np.array([1, 2, 3])
    result = _solve_cholesky(K, y)
    expected_result = np.linalg.solve(K, y)
    np.testing.assert_allclose(result, expected_result)


def test_jitter_kernel():
    K = np.array([[1, 0.5], [0.5, 1]])
    K_jitter = K.copy()
    eps = 1e-1
    _jitter_kernel(K_jitter, eps)
    expected_result = K + (eps * np.eye(K.shape[0]))
    np.testing.assert_allclose(K_jitter, expected_result)


def test_fit():
    # Test case 1: Example in documentation
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([5.0, 6.0])
    ls = 0.5
    eps = 1e-5
    result = _fit(X, y, ls)
    expected_result = np.array([5.0, 6.0])
    np.testing.assert_allclose(result, expected_result, atol=eps)


def test_predict():
    # Test case 1: Example in documentation
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    Xstar = np.array([[1.0, 1.0], [2.0, 2.0]])
    ls = 0.5
    a = np.array([5.0, 6.0])
    eps = 1e-6
    result = _predict(X, Xstar, ls, a)

    # Known, valid scikit-learn implementation
    gamma = 0.5 * (1.0 / ls**2)
    K = rbf_kernel(X, Xstar, gamma=gamma)
    expected_result = K.T @ a
    np.testing.assert_allclose(result, expected_result, atol=eps)


def test_mean_error_over_splits():
    # Test case 1: Example in documentation
    X = np.arange(10).reshape(-1, 1).astype(np.float64)
    rng = np.random.default_rng(17)
    y = rng.normal(size=(X.shape[0],))
    ls = 0.5
    mean_error = _mean_error_over_splits(X, y, ls, n_splits=3)
    expected_error = 0.7484052691169865

    np.testing.assert_almost_equal(mean_error, expected_error)


def test_find_best_ls():
    # Test case 1: Example in docstring
    X = np.arange(10).reshape(-1, 1).astype(np.float64)
    rng = np.random.default_rng(17)
    y = rng.normal(size=(X.shape[0],))
    ls_vals = np.array([0.5, 1.0])
    best_ls = _find_best_ls(X, y, ls_vals, n_splits=3)
    expected_best_ls = 0.5

    np.testing.assert_almost_equal(best_ls, expected_best_ls)


def test_detrend_gp():
    # Test case 1: Example in documentation
    X = np.arange(10).reshape(-1, 1).astype(np.float64)
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

    detrended_df = _detrend_all_signals_gp_numba(df, ls_vals, n_splits=3)

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
