import math
import random

import numba as nb
import numpy as np


@nb.njit("f8(f8[:])", fastmath=True)
def _mean(x: np.ndarray) -> float:
    n = x.size
    out = 0.0

    for i in range(n):
        out += x[i]

    out /= n
    return out


@nb.njit("i4(f8, f8)", fastmath=True)
def _unif_to_geom(u: float, p: float) -> int:
    return math.ceil(math.log(1 - u) / math.log(1 - p))


@nb.njit("f8(f8[:])", fastmath=True)
def _standard_error(x: np.ndarray) -> float:
    n = x.size
    se = 0.0
    xbar = _mean(x)

    for i in range(n):
        se += (x[i] - xbar) ** 2

    se *= 1.0 / n
    return math.sqrt(se)


@nb.njit
def _seed(a: int):
    random.seed(a)


@nb.njit("f8(f8[:], f8[:])", fastmath=True)
def _mean_squared_error(y: np.ndarray, yhat: np.ndarray) -> float:
    """
    Compute the mean squared error (MSE) between the true target values and
    the predicted values.

    Args:
        y (np.ndarray): Array of true target values.
        yhat (np.ndarray): Array of predicted values.

    Returns:
        float: Mean squared error between `y` and `yhat`.

    Examples:
        >>> y = np.array([1, 2, 3]).astype(np.float64)
        >>> yhat = np.array([1.5, 2.2, 2.8])
        >>> _mean_squared_error(y, yhat)
        0.11
    """
    n = y.size
    error = 0.0

    for i in range(n):
        error += (y[i] - yhat[i]) ** 2

    return (1.0 / n) * error


@nb.njit("f8[:, :](f8[:, :])", parallel=True, fastmath=True)
def _squared_euclidean_distance_xx(X: np.ndarray) -> np.ndarray:
    """
    Compute the squared Euclidean distance between all pairs of inputs in the array X.

    Args:
        X (np.ndarray): Input array of shape (n, d), where n is the number of
            inputs and d is the dimensionality.

    Returns:
        np.ndarray: The squared Euclidean distances between all pairs of inputs.
        The resulting array has shape (n, n), where element (i, j) represents
        the squared Euclidean distance between the i-th and j-th inputs in X.

    Example:
        >>> X = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float64)
        >>> distances = _squared_euclidean_distance_xx(X)
        >>> print(distances)
        array([[ 0,  8, 32],
               [ 8,  0,  8],
               [32,  8,  0]])
    """
    n = X.shape[0]  # Number of inputs

    distances = np.zeros((n, n), dtype=np.float64)

    for i in nb.prange(n):
        for j in nb.prange(i + 1, n):
            diff = X[i, :] - X[j, :]
            distances[i, j] = np.sum(diff * diff)
            distances[j, i] = distances[i, j]  # Symmetric, fill both entries

    return distances


@nb.njit("f8[:, :](f8[:, :], f8[:, :])", parallel=True, fastmath=True)
def _squared_euclidean_distance_xy(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute the squared Euclidean distance between pairs of inputs in arrays x and y.

    Args:
        X (np.ndarray): Input array of shape (n, d), where n is the number of
            inputs and d is the dimensionality.
        Y (np.ndarray): Input array of shape (m, d), where m is the number of
            inputs and d is the dimensionality.

    Returns:
        np.ndarray: The squared Euclidean distances between pairs of inputs.
        The resulting array has shape (n, m), where element (i, j) represents
        the squared Euclidean distance between the i-th input in X and the j-th input in Y.

    Example:
        >>> X = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float64)
        >>> Y = np.array([[2, 2], [4, 4]]).astype(np.float64)
        >>> distances_xy = _squared_euclidean_distance_xy(X, Y)
        >>> print(distances_xy)
        array([[ 1, 13],
               [ 5,  1],
               [25,  5]])
    """
    n = X.shape[0]  # Number of inputs in x
    m = Y.shape[0]  # Number of inputs in y

    distances = np.zeros((n, m), dtype=np.float64)

    for i in nb.prange(n):
        for j in nb.prange(m):
            diff = X[i, :] - Y[j, :]
            distances[i, j] = np.sum(diff * diff)

    return distances


@nb.njit("f8(f8[:], i4, f8)", fastmath=True)
def _sampen(x: np.ndarray, m: int, r: float) -> float:
    """
    Compute SampEn of inputs time series, `x` given a fixed m and r

    Args:
        x (np.ndarray): Time series signal. Shape is (n,)
        m (int): Embedding dimension
        r (float): Radial distance

    Returns:
        float: SampEn(x; m, r)
    """
    n = x.size
    run = np.zeros(n, dtype=np.int32)
    lastrun = np.zeros(n, dtype=np.int32)
    m += 1
    a = np.zeros(m, dtype=np.float64)
    b = np.zeros(m, dtype=np.float64)

    for i in range(n - 1):
        nj = n - i - 1
        x1 = x[i]

        for jj in range(nj):
            j = jj + i + 1

            if abs(x[j] - x1) < r:
                run[jj] = lastrun[jj] + 1

                # Increment up to the limit of m + 1 (recall we had m += 1 for runs)
                m1 = min(m, run[jj])
                for order in range(m1):
                    a[order] += 1.0

                    # Need like-to-like comparisons with A so have to account
                    # for boundary conditions
                    if j < n - 1:
                        b[order] += 1.0
            else:
                run[jj] = 0

        # Re-set the run counter for future iterations
        for j in range(nj):
            lastrun[j] = run[j]

    if a[-1] == 0.0:
        return np.nan
    else:
        return -math.log(a[m - 1] / b[m - 2])
