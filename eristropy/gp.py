import numba as nb
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import cho_factor, cho_solve
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from tqdm import tqdm

from eristropy.utils import (
    _mean_squared_error,
    _squared_euclidean_distance_xx,
    _squared_euclidean_distance_xy,
)


def _sklearn_fit_detrend_gp(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.RandomState,
    ls_range: tuple[float, float],
    scoring: str,
) -> np.ndarray:
    """
    Fits and predicts a Gaussian Process Regressor for a given signal, X,
    using a randomized search procedure over an RBF kernel with a valid time
    series cross-validation technique.

    Args:
        X (np.ndarray): Input signal.
        y (np.ndarray): Response vector.
        rng (np.random.RandomState): Pseudo-random number generator for reproducibility.
        ls_range (tuple[float, float]): Range of length scale values to search.
        scoring (str): Evaluation criteria for the random search procedure.
        params (StationarySignalParams): Dataclass defining relavant parameters
            for constructing stationary signals


    Returns:
        np.ndarray: Detrended signal from the final GP model.
    """

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    gp = GaussianProcessRegressor(
        kernel=RBF(length_scale_bounds="fixed"), normalize_y=True
    )

    fit_distns = {
        "kernel__length_scale": stats.uniform(
            loc=ls_range[0], scale=(ls_range[1] - ls_range[0])
        )
    }

    cv = RandomizedSearchCV(
        estimator=gp,
        param_distributions=fit_distns,
        scoring=scoring,
        cv=list(TimeSeriesSplit().split(X)),
        random_state=rng,
    )

    cv.fit(X, y)
    yhat = cv.predict(X)
    return y - yhat


def _detrend_all_signals_gp_sklearn(
    df: pd.DataFrame,
    signal_id: str,
    timestamp: str,
    value_col: str,
    rng: np.random.RandomState,
    ls_range: tuple[float, float],
    scoring: str,
) -> pd.DataFrame:
    """
    Detrends all signals in benchmark dataset using the randomized search
    GP approach implemented via Scikit-Learn

    Args:
        df (pd.DataFrame): The input DataFrame containing the signals.
        params (StationarySignalParams): Dataclass defining relavant parameters
            for constructing stationary signals

    Returns:
        pd.DataFrame: Set of detrended signals for all unique signal IDs
    """
    signal_ids = df[signal_id].unique()
    detrended_signals = []

    for sid in tqdm(signal_ids):
        X = np.arange(df.loc[df[signal_id] == sid].shape[0], dtype=np.float64).reshape(
            -1, 1
        )
        y = df.loc[df[signal_id] == sid, value_col].values.astype(np.float64)
        detrended_signal = _sklearn_fit_detrend_gp(X, y, rng, ls_range, scoring)

        n = detrended_signal.size
        tmp_df = pd.DataFrame(
            {
                signal_id: np.repeat(signal_id, n),
                timestamp: X.flatten(),
                value_col: detrended_signal,
            }
        )

        detrended_signals.append(tmp_df)

    out_df = pd.concat(detrended_signals, ignore_index=True)
    return out_df


@nb.njit("f8[:, :](f8[:, :], f8)")
def _rbf_kernel(D: np.ndarray, ls: float) -> np.ndarray:
    """
    Compute the radial basis function (RBF) kernel (covariance matrix) given the squared
    Euclidean distance matrix and the length scale `ls`.

    Args:
        D (np.ndarray): Squared Euclidean distance matrix of shape (n, m).
        ls (float): Length scale parameter for the RBF kernel.

    Returns:
        np.ndarray: The RBF kernel matrix of shape (n, m), where element (i, j) represents
        the RBF kernel value between the i-th and j-th inputs.

    Example:
        >>> D = np.array([[0, 8, 32],
                          [8, 0, 8],
                          [32, 8, 0]]).astype(np.float64)
        >>> ls = 0.5
        >>> kernel_matrix = _rbf_kernel(D, ls)
        >>> print(kernel_matrix)
        array([[1.00000000e+00, 1.12535175e-07, 1.60381089e-28],
               [1.12535175e-07, 1.00000000e+00, 1.12535175e-07],
               [1.60381089e-28, 1.12535175e-07, 1.00000000e+00]])
    """
    gamma = 0.5 * (1.0 / ls**2)
    K = -gamma * D
    return np.exp(K)


@nb.njit
def _time_series_split(
    X: np.ndarray, n_splits: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate train-test splits for a time series dataset.

    Args:
        X (np.ndarray): Input array-like object representing the time series data.
        n_splits (int, optional): Number of splits to generate (default: 5).

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: A list of tuples, where each tuple contains
        the train and test indices for a split. Each train-test split is represented as
        (train_indices, test_indices).

    Example:
        >>> X = np.arange(6)
        >>> splits = time_series_split(x, n_splits=3)
        >>> for train_indices, test_indices in splits:
        ...     print(f"Train indices: {train_indices}, Test indices: {test_indices}")
        ...
        Train indices: [0 1 2], Test indices: [3]
        Train indices: [0 1 2 3], Test indices: [4]
        Train indices: [0 1 2 3 4], Test indices: [5]
    """
    n = X.shape[0]
    indices = np.arange(n)
    test_size = n // (n_splits + 1)
    test_starts = range(n - n_splits * test_size, n, test_size)

    splits = []
    for test_start in test_starts:
        train_indices = indices[:test_start]
        test_indices = indices[test_start : test_start + test_size]
        splits.append((train_indices, test_indices))

    return splits


def _solve_cholesky(K: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Solve the linear equation system Kx = y using the Cholesky factorization.

    Parameters:
        K (np.ndarray): The positive definite matrix K of shape (n, n).
        y (np.ndarray): The right-hand side vector y of shape (n,).

    Returns:
        np.ndarray: The solution vector x of shape (n,).

    Notes:
        - The input matrix K should be positive definite.
        - The dimensions of K and y should be compatible for matrix-vector multiplication.

    Examples:
        >>> K = np.array([[1, 1/2, 0], [1/2, 1, 1/3], [0, 1/3, 1]])
        >>> y = np.array([1, 2, 3])
        >>> _solve_cholesky(K, y)
        array([0.60869565, 0.7826087, 2.73913043])
    """
    L = cho_factor(K, lower=True, check_finite=False)[0]
    a = cho_solve((L, True), y, check_finite=False)
    return a


@nb.njit("(f8[:, :], f8)", fastmath=True)
def _jitter_kernel(K: np.ndarray, eps: float = 1e-6) -> None:
    """
    Apply jittering to the kernel matrix K to ensure numerical stability.

    Parameters:
        K (np.ndarray): The kernel matrix K of shape (n, n).
        eps (float, optional): The jittering constant to add to the diagonal elements.
            Defaults to 1e-6.

    Returns:
        None

    Notes:
        - The input matrix K is modified in place.

    Examples:
        >>> K = np.array([[1, 0.5], [0.5, 1]])
        >>> eps = 1e-1
        >>> _jitter_kernel(K, eps)
        >>> K
        array([[1.1, 0.5],
               [0.5, 1.1]])
    """
    n = K.shape[0]

    for i in range(n):
        K[i, i] += eps


def _fit(
    X: np.ndarray,
    y: np.ndarray,
    ls: float,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a Gaussian Process (GP) model to the training data.

    Args:
        X (np.ndarray): Array of training input features. Shape: (n, d).
        y (np.ndarray): Array of training target values. Shape: (n,).
        ls (float): Length scale parameter for the RBF kernel.
        eps (float, optional): Jitter value for numerical stability. Defaults to 1e-6.

    Returns:
        np.ndarray: Array of coefficients (alpha) obtained from solving the linear system
        in the GP model. Shape: (n,).

    Notes:
        The function fits a GP model to the training data using the squared Euclidean
        distance between input features. It computes the RBF kernel matrix
        based on the distance and adds jitter (small value on the diagonal) for numerical
        stability. It then solves the linear system Ka = y, where `K` is the kernel
        matrix and `a` are the coefficients to be determined.

    Examples:
        >>> X = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> y = np.array([5.0, 6.0])
        >>> ls = 0.5
        >>> a = _fit(X, y, ls)
        >>> a
        array([5.0, 6.0])
    """
    D = _squared_euclidean_distance_xx(X)
    K = _rbf_kernel(D, ls)
    _jitter_kernel(K, eps)
    a = _solve_cholesky(K, y)
    return a


def _predict(X: np.ndarray, Xstar: np.ndarray, ls: float, a: np.ndarray) -> np.ndarray:
    """
    Predict the target values of a Gaussian Process (GP) model for the given input features.

    Args:
        X (np.ndarray): Array of training input features. Shape: (n, d).
        Xstar (np.ndarray): Array of input features for prediction. Shape: (m, d).
        ls (float): Length scale parameter for the RBF kernel.
        a (np.ndarray): Array of coefficients obtained from fitting the GP model. Shape: (n,).

    Returns:
        np.ndarray: Array of predicted target values. Shape: (m,).

    Notes:
        The function predicts the target values for the input features `Xstar` based on a
        trained GP model. It computes the squared Euclidean distance between `X` and `Xstar`,
        and then evaluates the RBF kernel using the distance and length scale. The predicted
        target values are obtained by matrix-multiplying the kernel matrix with the coefficients
        obtained during training.

    Examples:
        >>> X = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> Xstar = np.array([[1.0, 1.0], [2.0, 2.0]])
        >>> ls = 0.5
        >>> a = np.array([5.0, 6.0])
        >>> _predict(X, Xstar, ls, a)
        array([1.48868812e+00, 2.72399604e-04])
    """
    D = _squared_euclidean_distance_xy(X, Xstar)
    K = _rbf_kernel(D, ls)
    return K.T @ a


def _partition_data(
    X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray
) -> tuple[np.ndarray, ...]:
    """
    Partition the input features X and target values y into training and testing sets.

    Parameters:
        X (np.ndarray): The input feature matrix of shape (n_samples, n_features).
        y (np.ndarray): The target values vector of shape (n_samples,).
        train_idx (np.ndarray): The indices of the samples for the training set.
        test_idx (np.ndarray): The indices of the samples for the testing set.

    Returns:
        tuple[np.ndarray, ...]: A tuple containing Xtrain, ytrain, Xtest, and ytest.

    Notes:
        - The input feature matrix X and target values y should have compatible dimensions.
        - The train_idx and test_idx should be valid indices corresponding to the samples.
    """
    Xtrain = X[train_idx, :]
    ytrain = y[train_idx]
    Xtest = X[test_idx, :]
    ytest = y[test_idx]
    return Xtrain, ytrain, Xtest, ytest


def _mean_error_over_splits(
    X: np.ndarray,
    y: np.ndarray,
    ls: float,
    n_splits: int = 5,
    eps: float = 1e-6,
) -> float:
    """
    Estimate the average mean squared error (MSE) across the validation splits
    for a given length scale (ls) value in the radial basis function (RBF) kernel.

    Args:
        X (np.ndarray): Input features of shape (n, d), where n is the number of
            samples and d is the dimensionality.
        y (np.ndarray): Target values of shape (n,) corresponding to the input features.
        ls (float): Length scale parameter for the RBF kernel.
        n_splits (int, optional): Number of splits to generate (default: 5).
        eps (float, optional): Small value added to the kernel matrix for numerical stability
            (default: 1e-6).

    Returns:
        float: The average mean squared error (MSE) across the validation splits.

    Examples:
        >>> X = np.arange(10).reshape(-1, 1).astype(np.float64)
        >>> rng = np.random.default_rng(17)
        >>> y = rng.normal(size=(X.shape[0],))
        >>> ls = 0.5
        >>> mean_error = _mean_error_over_splits(X, y, ls, n_splits=3)
        >>> mean_error
        0.7484052691169865
    """
    splits = _time_series_split(X, n_splits)
    errors = np.zeros((n_splits,), dtype=np.float64)

    for i in range(n_splits):
        train_idx, test_idx = splits[i]
        Xtrain, ytrain, Xtest, ytest = _partition_data(X, y, train_idx, test_idx)
        a = _fit(Xtrain, ytrain, ls, eps)
        yhat = _predict(Xtrain, Xtest, ls=ls, a=a)
        errors[i] = _mean_squared_error(ytest, yhat)

    return np.mean(errors)


def _find_best_ls(
    X: np.ndarray,
    y: np.ndarray,
    ls_vals: np.ndarray,
    n_splits: int = 5,
    eps: float = 1e-6,
) -> float:
    """
    Find the best length scale (ls) value for a Gaussian Process (GP) model by searching
    over a range of ls values and identifying the one with the minimal validation error.

    Args:
        X (np.ndarray): Input features of shape (n, d), where n is the number of samples
            and d is the dimensionality.
        y (np.ndarray): Target values of shape (n,) corresponding to the input features.
        ls_vals (np.ndarray): Array of ls values to search over.
        n_splits (int, optional): Number of splits to generate for cross-validation (default: 5).
        eps (float, optional): Small value added to the kernel matrix for numerical stability
            (default: 1e-6).

    Returns:
        float: The ls value that corresponds to the minimal validation error.

    Examples:
        >>> X = np.arange(10).reshape(-1, 1).astype(np.float64)
        >>> rng = np.random.default_rng(17)
        >>> y = rng.normal(size=(X.shape[0],))
        >>> ls_vals = np.array([0.5, 1.0])
        >>> best_ls = _find_best_ls(X, y, ls_vals, n_splits=3)
        >>> best_ls
        0.5
    """
    n = ls_vals.size
    errors = np.zeros(n, dtype=np.float64)

    for i in range(n):
        errors[i] = _mean_error_over_splits(
            X, y, ls=ls_vals[i], n_splits=n_splits, eps=eps
        )

    ls_star = ls_vals[np.argmin(errors)]
    return ls_star


def _detrend_gp(
    X: np.ndarray,
    y: np.ndarray,
    ls_vals: np.ndarray,
    n_splits: int = 5,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Detrend a signal using a Gaussian Process (GP) model and finding the best length scale (ls) value via cross-validation.

    Args:
        X (np.ndarray): Input features of shape (n, d), where n is the number of samples
            and d is the dimensionality.
        y (np.ndarray): Target values of shape (n,) corresponding to the input features.
        ls_vals (np.ndarray): Array of ls values to search over during cross-validation.
        n_splits (int, optional): Number of splits to generate for cross-validation (default: 5).
        eps (float, optional): Small value added to the kernel matrix for numerical stability
            (default: 1e-6).

    Returns:
        np.ndarray: The detrended signal obtained by subtracting the predicted signal
        from the original signal.

    Examples:
        >>> import numpy as np
        >>> X = np.arange(10).reshape(-1, 1).astype(np.float64)
        >>> rng = np.random.default_rng(17)
        >>> y = rng.normal(size=(X.shape[0],))
        >>> ls_vals = np.array([0.5, 1.0])
        >>> detrended_signal = _detrend_gp(X, y, ls_vals, n_splits=3)
        >>> detrended_signal
        array([ 1.06695763e-06,  2.54575510e-07, -4.44978049e-07, -9.54630765e-07,
               -1.81473399e-06,  3.67333456e-07, -7.57561006e-07, -7.54210002e-07,
               -1.14763661e-07, -3.60613987e-08])

    """
    ls_star = _find_best_ls(X, y, ls_vals=ls_vals, n_splits=n_splits, eps=eps)
    a = _fit(X, y, ls=ls_star, eps=eps)
    yhat = _predict(X, X, ls=ls_star, a=a)
    return y - yhat


def _detrend_all_signals_gp_numba(
    df: pd.DataFrame,
    signal_id: str,
    timestamp: str,
    value_col: str,
    rng: np.random.RandomState,
    ls_range: tuple[float, float],
    n_searches: int,
    n_splits: int,
    eps: float,
) -> pd.DataFrame:
    """
    Detrends all signals in a DataFrame using a Gaussian Process (GP) approach.

    Args:
        df (pd.DataFrame): Input DataFrame containing signal observations.
        params (StationarySignalParams): Dataclass defining relavant parameters
            for constructing stationary signals

    Returns:
        pd.DataFrame: DataFrame containing the detrended signals.

    Example:
        >>> df = pd.DataFrame({
        ...     "signal_id": ["abc", "abc", "abc", "abc", "abc", "def", "def", "def", "def", "def"],
        ...     "timestamp": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        ...     "value": [10, 12, 15, 17, 18, 5, 6, 7, 11, 14]
        ... })
        >>> ls_vals = np.array([0.5, 1.0])
        >>> detrended_df = _detrend_all_signals_gp(df, ls_vals, n_splits=3)
        >>> detrended_df
        signal_id  timestamp         value
        0       abc        1.0  9.138937e-06
        1       abc        2.0 -1.467807e-06
        2       abc        3.0  1.298148e-05
        3       abc        4.0 -1.006977e-06
        4       abc        5.0  1.686713e-05
        5       def        1.0  3.581833e-06
        6       def        2.0  1.587954e-06
        7       def        3.0  3.257620e-06
        8       def        4.0  8.809471e-07
        9       def        5.0  1.300595e-05
    """
    # Sort the DataFrame by timestamp
    df = df.sort_values(timestamp)

    # Group the DataFrame by signal ID
    grouped = df.groupby(signal_id)

    # Initialize an empty list to store the detrended signals
    detrended_signals = []

    # Get the ls search values
    ls_vals = rng.uniform(ls_range[0], ls_range[1], size=n_searches)

    # Iterate over each group and compute the detrended signal
    for _, group in tqdm(grouped):
        # Detrend the signal via linear regression
        X = np.arange(group.shape[0], dtype=np.float64).reshape(-1, 1)
        y = group[value_col].values.astype(np.float64)
        detrended_values = _detrend_gp(X, y, ls_vals, n_splits=n_splits, eps=eps)

        # Create a new DataFrame for the detrended signal
        detrended_df = pd.DataFrame(
            {
                signal_id: group[signal_id].values,
                timestamp: X.flatten(),
                value_col: detrended_values,
            }
        )

        # Append the detrended signal DataFrame to the list
        detrended_signals.append(detrended_df)

    # Concatenate all the detrended signals into a single DataFrame
    detrended_df = pd.concat(detrended_signals, ignore_index=True)

    return detrended_df
