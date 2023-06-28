import numba as nb
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.multitest import multipletests


def _difference(x: np.ndarray) -> np.ndarray:
    """
    Compute the differenced signal to make a time series statistically stationary.

    Args:
        x (np.ndarray): The input time series signal.

    Returns:
        np.ndarray: The differenced signal.

    Example:
        >>> x = [1, 3, 6, 10, 15]
        >>> difference(x)
        array([2, 3, 4, 5])
    """
    if len(x) < 2:
        raise ValueError("Input must have at least two elements.")

    if np.isnan(x).any() or np.isinf(x).any():
        raise ValueError("Input contains NaN or np.inf values.")

    return np.diff(x)


def _difference_all_signals(
    df: pd.DataFrame,
    signal_id: str = "signal_id",
    timestamp: str = "timestamp",
    value_col: str = "value",
) -> pd.DataFrame:
    """
    Compute the differenced signals for each unique signal ID in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing signal observations.
        signal_id (str, optional): The column name for the signal ID. Defaults to "signal_id".
        timestamp (str, optional): The column name for the timestamp. Defaults to "timestamp".
        value_col (str, optional): The column name for the signal values. Defaults to "value".

    Returns:
        pd.DataFrame: The differenced signals for each unique signal ID.

    Example:
        >>> df = pd.DataFrame({"signal_id": ["abc", "abc", "def", "def"],
                              "timestamp": [1, 2, 1, 2],
                              "value": [2, 3, 5, 7]})
        >>> difference_all_signals(df)
           signal_id  timestamp       value
        0        abc          2           1
        1        def          2           2
    """
    df = df.sort_values(timestamp)

    # Group the DataFrame by signal ID
    grouped = df.groupby(signal_id)

    # Initialize an empty list to store the differenced signals
    diff_signals = []

    # Iterate over each group and compute the differenced signal
    for _, group in grouped:
        # Compute the differenced signal using np.diff
        x = group[value_col].values
        diff_values = _difference(x)

        # Create a new DataFrame for the differenced signal
        diff_df = pd.DataFrame(
            {
                signal_id: [group[signal_id].iloc[-1]] * len(diff_values),
                timestamp: group[timestamp].iloc[1:],
                "value": diff_values,
            }
        )

        # Append the differenced signal DataFrame to the list
        diff_signals.append(diff_df)

    # Concatenate all the differenced signals into a single DataFrame
    diff_df = pd.concat(diff_signals, ignore_index=True)

    return diff_df


def _detrend_linreg(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Detrends a signal using linear regression.

    Args:
        X (np.ndarray): Input feature for linear regression. Shape: (n, d).
        y (np.ndarray): Target signal to be detrended. Shape: (n,).

    Returns:
        np.ndarray: The detrended signal.

    Examples:
        >>> import numpy as np
        >>> X = np.array([1, 2, 3, 4, 5])
        >>> y = np.array([2, 3, 5, 7, 8])
        >>> _detrend_linreg(X, y)
        array([0.2, -0.4, 0.0, 0.4, -0.2])

    Notes:
        This function performs linear regression between the input features `X` and
        the target signal `y` to estimate the linear trend. It subtracts the estimated
        trend from the original signal `y` to obtain the detrended signal.
    """
    Xint = np.column_stack((X, np.ones((X.shape[0], 1))))
    beta = np.linalg.lstsq(Xint, y, rcond=None)[0]
    yhat = Xint @ beta
    return y - yhat


def _detrend_all_signals_linreg(
    df: pd.DataFrame,
    signal_id: str = "signal_id",
    timestamp: str = "timestamp",
    value_col: str = "value",
) -> pd.DataFrame:
    """
    Compute the detrended signals via linear regression for each unique signal ID in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing signal observations.
        signal_id (str): The column name for the signal ID.
        timestamp (str): The column name for the timestamp.
        value_col (str): The column name for the signal values.

    Returns:
        pd.DataFrame: The detrended signals for each unique signal ID.

    Example:
        >>> df = pd.DataFrame({"signal_id": ["abc", "abc", "def", "def"],
                              "timestamp": [1, 2, 1, 2],
                              "value": [2, 3, 5, 7]})
        >>> _detrend_all_signals_linreg(df)
           signal_id  timestamp  value
        0        abc          1    0.0
        1        abc          2    0.0
        2        def          1    0.0
        3        def          2    0.0
    """
    # Sort the DataFrame by timestamp
    df = df.sort_values(timestamp)

    # Group the DataFrame by signal ID
    grouped = df.groupby(signal_id)

    # Initialize an empty list to store the detrended signals
    detrended_signals = []

    # Iterate over each group and compute the detrended signal
    for _, group in grouped:
        # Detrend the signal via linear regression
        X = group[timestamp].values.reshape(-1, 1)
        y = group[value_col].values
        detrended_values = _detrend_linreg(X, y)

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


@nb.njit(parallel=True)
def _squared_euclidean_distance_xx(X: np.ndarray) -> np.ndarray:
    """
    Compute the squared Euclidean distance between all pairs of inputs in the array X.

    Args:
        X (np.ndarray): Input array of shape (n, d), where n is the number of inputs and d is the dimensionality.

    Returns:
        np.ndarray: The squared Euclidean distances between all pairs of inputs.
        The resulting array has shape (n, n), where element (i, j) represents the squared Euclidean distance
        between the i-th and j-th inputs in X.

    Example:
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> distances = _squared_euclidean_distance_xx(X)
        >>> print(distances)
        array([[ 0,  8, 32],
               [ 8,  0,  8],
               [32,  8,  0]])
    """
    n = X.shape[0]  # Number of inputs

    distances = np.zeros((n, n), dtype=np.float64)  # Initialize the distances array

    for i in nb.prange(n):
        for j in nb.prange(i + 1, n):
            diff = X[i, :] - X[j, :]
            distances[i, j] = np.sum(diff * diff)
            distances[j, i] = distances[i, j]  # Symmetric, fill both entries

    return distances


@nb.njit(parallel=True)
def _squared_euclidean_distance_xy(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute the squared Euclidean distance between pairs of inputs in arrays x and y.

    Args:
        X (np.ndarray): Input array of shape (n, d), where n is the number of inputs and d is the dimensionality.
        Y (np.ndarray): Input array of shape (m, d), where m is the number of inputs and d is the dimensionality.

    Returns:
        np.ndarray: The squared Euclidean distances between pairs of inputs.
        The resulting array has shape (n, m), where element (i, j) represents the squared Euclidean distance
        between the i-th input in X and the j-th input in Y.

    Example:
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> Y = np.array([[2, 2], [4, 4]])
        >>> distances_xy = _squared_euclidean_distance_xy(X, Y)
        >>> print(distances_xy)
        array([[ 1, 13],
               [ 5,  1],
               [25,  5]])
    """
    n = X.shape[0]  # Number of inputs in x
    m = Y.shape[0]  # Number of inputs in y

    distances = np.zeros((n, m), dtype=np.float64)  # Initialize the distances array

    for i in nb.prange(n):
        for j in nb.prange(m):
            diff = X[i, :] - Y[j, :]
            distances[i, j] = np.sum(diff * diff)

    return distances


@nb.njit
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
                          [32, 8, 0]])
        >>> ls = 0.5
        >>> kernel_matrix = _rbf_kernel(D, ls)
        >>> print(kernel_matrix)
        array([[1.00000000e+00, 1.12535175e-07, 1.60381089e-28],
               [1.12535175e-07, 1.00000000e+00, 1.12535175e-07],
               [1.60381089e-28, 1.12535175e-07, 1.00000000e+00]])
    """
    K = -1 / (2 * (ls**2)) * D
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


@nb.njit
def _mean_squared_error(y: np.ndarray, yhat: np.ndarray) -> float:
    """
    Compute the mean squared error (MSE) between the true target values and the predicted values.

    Args:
        y (np.ndarray): Array of true target values.
        yhat (np.ndarray): Array of predicted values.

    Returns:
        float: Mean squared error between `y` and `yhat`.

    Examples:
        >>> y = np.array([1, 2, 3])
        >>> yhat = np.array([1.5, 2.2, 2.8])
        >>> _mean_squared_error(y, yhat)
        0.11
    """
    n = y.size
    error = 0.0

    for i in range(n):
        error += (y[i] - yhat[i]) ** 2

    return error / n


@nb.njit
def _fit(X: np.ndarray, y: np.ndarray, ls: float, eps: float = 1e-6) -> np.ndarray:
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
        distance between input features. It computes the RBF kernel matrix (Gram matrix)
        based on the distance and adds jitter (small value on the diagonal) for numerical
        stability. It then solves the linear system K @ alpha = y, where K is the kernel
        matrix and alpha are the coefficients to be determined.

    Examples:
        >>> X = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> y = np.array([5.0, 6.0])
        >>> ls = 0.5
        >>> alpha = _fit(X, y, ls)
        >>> alpha
        array([5.0, 6.0])
    """
    D = _squared_euclidean_distance_xx(X)
    K = _rbf_kernel(D, ls)
    K += eps * np.eye(K.shape[0])  # Adds jitter for numerical stability
    alpha = np.linalg.solve(K, y)
    return alpha


@nb.njit
def _predict(
    X: np.ndarray, Xstar: np.ndarray, ls: float, alpha: np.ndarray
) -> np.ndarray:
    """
    Predict the target values of a Gaussian Process (GP) model for the given input features.

    Args:
        X (np.ndarray): Array of training input features. Shape: (n, d).
        Xstar (np.ndarray): Array of input features for prediction. Shape: (m, d).
        ls (float): Length scale parameter for the RBF kernel.
        alpha (np.ndarray): Array of coefficients obtained from fitting the GP model. Shape: (n,).

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
        >>> alpha = np.array([5.0, 6.0])
        >>> _predict(X, Xstar, ls, alpha)
        array([1.48868812e+00, 2.72399604e-04])
    """
    D = _squared_euclidean_distance_xy(X, Xstar)
    K = _rbf_kernel(D, ls)
    return K.T @ alpha


@nb.njit
def _mean_error_over_splits(
    X: np.ndarray, y: np.ndarray, ls: float, n_splits: int = 5, eps: float = 1e-6
) -> float:
    """
    Estimate the average mean squared error (MSE) across the validation splits
    for a given length scale (ls) value in the radial basis function (RBF) kernel.

    Args:
        X (np.ndarray): Input features of shape (n, d), where n is the number of samples and d is the dimensionality.
        y (np.ndarray): Target values of shape (n,) corresponding to the input features.
        ls (float): Length scale parameter for the RBF kernel.
        n_splits (int, optional): Number of splits to generate (default: 5).
        eps (float, optional): Small value added to the kernel matrix for numerical stability (default: 1e-6).

    Returns:
        float: The average mean squared error (MSE) across the validation splits.

    Examples:
        >>> X = np.arange(10).reshape(-1, 1)
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
        Xtrain = X[train_idx, :]
        ytrain = y[train_idx]
        Xtest = X[test_idx, :]
        ytest = y[test_idx]

        alpha = _fit(Xtrain, ytrain, ls=ls, eps=eps)
        yhat = _predict(Xtrain, Xtest, ls=ls, alpha=alpha)
        errors[i] = _mean_squared_error(ytest, yhat)

    return np.mean(errors)


@nb.njit
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
        X (np.ndarray): Input features of shape (n, d), where n is the number of samples and d is the dimensionality.
        y (np.ndarray): Target values of shape (n,) corresponding to the input features.
        ls_vals (np.ndarray): Array of ls values to search over.
        n_splits (int, optional): Number of splits to generate for cross-validation (default: 5).
        eps (float, optional): Small value added to the kernel matrix for numerical stability (default: 1e-6).

    Returns:
        float: The ls value that corresponds to the minimal validation error.

    Examples:
        >>> X = np.arange(10).reshape(-1, 1)
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


@nb.njit
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
        X (np.ndarray): Input features of shape (n, d), where n is the number of samples and d is the dimensionality.
        y (np.ndarray): Target values of shape (n,) corresponding to the input features.
        ls_vals (np.ndarray): Array of ls values to search over during cross-validation.
        n_splits (int, optional): Number of splits to generate for cross-validation (default: 5).
        eps (float, optional): Small value added to the kernel matrix for numerical stability (default: 1e-6).

    Returns:
        np.ndarray: The detrended signal obtained by subtracting the predicted signal from the original signal.

    Examples:
        >>> import numpy as np
        >>> X = np.arange(10).reshape(-1, 1)
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
    alpha = _fit(X, y, ls=ls_star, eps=eps)
    yhat = _predict(X, X, ls=ls_star, alpha=alpha)
    return y - yhat


def _detrend_all_signals_gp(
    df: pd.DataFrame,
    ls_vals: np.ndarray,
    signal_id: str = "signal_id",
    timestamp: str = "timestamp",
    value_col: str = "value",
    n_splits: int = 5,
    eps: float = 1e-6,
) -> pd.DataFrame:
    """
    Detrends all signals in a DataFrame using a Gaussian Process (GP) approach.

    Args:
        df (pd.DataFrame): Input DataFrame containing signal observations.
        ls_vals (np.ndarray): Array of ls values to search over during cross-validation.
        signal_id (str): Column name for the signal ID (default: "signal_id").
        timestamp (str): Column name for the timestamp (default: "timestamp").
        value_col (str): Column name for the signal values (default: "value").
        n_splits (int, optional): Number of splits to generate for cross-validation (default: 5).
        eps (float, optional): Small value added to the kernel matrix for numerical stability (default: 1e-6).

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

    # Iterate over each group and compute the detrended signal
    for _, group in grouped:
        # Detrend the signal via linear regression
        X = group[timestamp].values.reshape(-1, 1).astype(np.float64)
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


def _calculate_pvalues(
    df: pd.DataFrame,
    signal_id: str = "signal_id",
    timestamp: str = "timestamp",
    value_col: str = "value",
) -> np.ndarray:
    """
    Calculate the p-values for each unique signal ID in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing signal observations.
        signal_id (str, optional): Column name for the signal ID (default: 'signal_id').
        timestamp (str, optional): Column name for the timestamp (default: 'timestamp').
        value_col (str, optional): Column name for the signal values (default: 'value').

    Returns:
        np.ndarray: Array containing the computed p-values.

    Notes:
        This function uses the augmented Dickey-Fuller test from the statsmodels.tsa.stattools module
        to compute the p-values for each unique signal ID in the DataFrame. If a signal is not long enough
        to estimate the p-value, a p-value of 1.0 is assigned.

    Example:
        >>> signal_ids = np.repeat(["abc", "def"], 100)
        >>> timestamps = np.tile(np.arange(100), 2)
        >>> abc_values = np.linspace(0, 100, 100)
        >>> def_values = np.sin(np.linspace(0, 2 * np.pi, 100))
        >>> values = np.concatenate((abc_values, def_values))
        >>> df = pd.DataFrame({
        ...     "signal_id": signal_ids,
        ...     "timestamp": timestamps,
        ...     "value": values
        ... })
        >>> pvalues = _calculate_pvalues(df)
        >>> pvalues
        array([0.9134984832798951, 0.0])
    """
    # Sort the DataFrame by timestamp
    df = df.sort_values(timestamp)

    # Group the DataFrame by signal ID
    grouped = df.groupby(signal_id)

    pvalues = []

    for _, group in grouped:
        y = group[value_col].values.astype(np.float64)
        try:
            pvalue = adfuller(y)[1]
        except ValueError as e:
            # Handle the case where the signal is (probably) not long enough
            pvalue = 1.0
            print(f"An error occurred for group: {group[signal_id].iloc[0]}")
            print(f"Error message: {str(e)}")
        pvalues.append(pvalue)

    return np.asarray(pvalues)


def determine_stationary_signals(
    df: pd.DataFrame,
    alpha: float = 0.05,
    signal_id: str = "signal_id",
    timestamp: str = "timestamp",
    value_col: str = "value",
) -> tuple[float, np.ndarray]:
    """
    Compute the fraction of signals in the DataFrame that are statistically stationary.

    This function calculates the fraction of signals in the input DataFrame that exhibit
    stationarity using the Augmented Dickey-Fuller test. It groups the DataFrame by signal ID
    and performs the test for each group. The fraction is determined based on the proportion
    of signals that reject the null hypothesis of non-stationarity at the specified significance level.

    Args:
        df (pd.DataFrame): The input DataFrame containing signal observations.
        alpha (float, optional): The significance level for the stationarity test (default: 0.05).
        signal_id (str, optional): The column name for the signal ID (default: 'signal_id').
        timestamp (str, optional): The column name for the timestamp (default: 'timestamp').
        value_col (str, optional): The column name for the signal values (default: 'value').

    Returns:
        tuple[float, np.ndarray]: The fraction of signals that are statistically stationary
        and array of signal IDs which are statistically stationary

    Raises:
        ValueError: If the required columns are missing in the DataFrame,
                    if the DataFrame is empty, if the significance level is invalid,
                    if the signal is not long enough for the ADF test,
                    or if no groups are found after grouping by signal ID.

    Example:
        >>> signal_ids = np.repeat(["abc", "def"], 100)
        >>> timestamps = np.tile(np.arange(100), 2)
        >>> abc_values = np.linspace(0, 100, 100)
        >>> def_values = np.sin(np.linspace(0, 2 * np.pi, 100))
        >>> values = np.concatenate((abc_values, def_values))
        >>> df = pd.DataFrame({
        ...     "signal_id": signal_ids,
        ...     "timestamp": timestamps,
        ...     "value": values
        ... })
        >>> stationary_fraction, stationary_signals = determine_stationary_signals(df)
        >>> fraction_stationary
        0.5, array(["def"])

    """
    # Check if the required columns are present in the DataFrame
    required_columns = [signal_id, timestamp, value_col]
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Check if the DataFrame is empty
    if df.empty:
        raise ValueError("Input DataFrame is empty")

    # Check if the significance level is within the valid range
    if not 0 < alpha < 1:
        raise ValueError("Significance level must be between 0 and 1 (exclusive)")

    unique_signal_ids = df.signal_id.unique()

    # Check if no groups are found
    if len(unique_signal_ids) == 0:
        raise ValueError("No unique signal IDs")

    pvalues = _calculate_pvalues(df, signal_id, timestamp, value_col)
    multi_rejects = multipletests(pvalues, alpha=alpha)[0]
    return multi_rejects.mean(), unique_signal_ids[multi_rejects]


def make_stationary_signals(
    df: pd.DataFrame,
    method: str,
    detrend_type: str = "gp",
    signal_id: str = "signal_id",
    timestamp: str = "timestamp",
    value_col: str = "value",
    alpha: float = 0.05,
    random_seed: int = None,
    ls_range: tuple[float, float] = (1.0, 100.0),
    n_searches: int = 10,
    n_splits: int = 5,
    eps: float = 1e-6,
) -> pd.DataFrame:
    """
    Create stationary signals using the specified method.

    Args:
        df (pd.DataFrame): The input DataFrame containing the signals.
        method (str): The method to make the signals stationary. Valid options are 'difference' and 'detrend'.
        detrend_type (str): The type of de-trending. Required when method is 'detrend'. Valid options are 'lr' and 'gp'.
        signal_id (str): The column name for the signal ID. Default is 'signal_id'.
        timestamp (str): The column name for the timestamp. Default is 'timestamp'.
        value_col (str): The column name for the signal values. Default is 'value'.
        alpha (float): The significance level for the stationarity test. Must be in the range (0, 1). Default is 0.05.
        random_seed (int): The random seed for generating ls_vals. Must be an integer. Default is None.
        ls_range (tuple[float, float]): The range of ls values for GP detrending. The lower bound must be greater than 0. Default is (1.0, 100.0).
        n_searches (int): The number of ls values to consider for GP detrending. Must be a positive integer. Default is 10.
        n_splits (int): The number of cross-validation splits. Must be a positive integer. Default is 5.
        eps (float, optional): Small value added to the kernel matrix for numerical stability (default: 1e-6).

    Returns:
        pd.DataFrame: The DataFrame with stationary signals.

    Raises:
        TypeError: If the input is not a pandas DataFrame.
        ValueError: If the input DataFrame is empty, if the input columns are missing, if the method or detrend_type values are invalid,
                    if the alpha value is not within the valid range, if the ls_range lower bound is not greater than 0,
                    if n_searches or n_splits are not positive integers.

    Example:
        >>> signal_ids = np.repeat(["abc", "def"], 100)
        >>> timestamps = np.tile(np.arange(100), 2)
        >>> rng = np.random.default_rng(17)
        >>> abc_values = rng.uniform(-5, 5, size=(100,))
        >>> def_values = rng.uniform(-5, 5, size=(100,))
        >>> values = np.concatenate((abc_values, def_values))
        >>> df = pd.DataFrame({
        ...     "signal_id": signal_ids,
        ...     "timestamp": timestamps,
        ...     "value": values
        ... })
        >>> make_stationary_signals(df, method="difference")
            signal_id  timestamp     value
        0         abc          1 -6.841017
        1         abc          2  3.967715
        2         abc          3 -1.896646
        3         abc          4 -1.531380
        4         abc          5  1.708821
        ..        ...        ...       ...
        193       def         95  0.653840
        194       def         96  0.846767
        195       def         97  5.441443
        196       def         98 -8.955780
        197       def         99  5.397502

    """

    # Check if the input is a pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    # Check if the DataFrame is empty
    if df.empty:
        raise ValueError("Input DataFrame is empty")

    # Check if the required columns are present in the DataFrame
    required_columns = [signal_id, timestamp, value_col]
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Convert 'timestamp' and 'value_col' columns to numeric
    df[timestamp] = pd.to_numeric(df[timestamp], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    # Check if the DataFrame contains NaN or np.inf values
    if (
        df[[timestamp, value_col]].isnull().values.any()
        or np.isinf(df[[timestamp, value_col]].values).any()
    ):
        raise ValueError(
            "Input DataFrame contains NaN or np.inf values or non-numeric data."
        )

    # Validate the method
    if method not in ["difference", "detrend"]:
        raise ValueError(
            "Invalid value for 'method'. Must be either 'difference' or 'detrend'."
        )

    # Validate detrend_type when method is 'detrend'
    if method == "detrend":
        if detrend_type not in ["lr", "gp"]:
            raise ValueError(
                "Invalid value for 'detrend_type'. Must be either 'lr' or 'gp'."
            )

    # Validate the significance level (alpha)
    if not (0 < alpha < 1):
        raise ValueError("Significance level must be between 0 and 1 (exclusive).")

    # Validate the lower bound of ls_range
    ls_lower_bound = ls_range[0]
    if ls_lower_bound <= 0:
        raise ValueError("The lower bound of 'ls_range' must be greater than 0.")

    # Validate n_searches
    if not isinstance(n_searches, int) or n_searches <= 0:
        raise ValueError(
            "The number of searches (n_searches) must be a positive integer."
        )

    # Validate n_splits
    if not isinstance(n_splits, int) or n_splits <= 0:
        raise ValueError("The number of splits (n_splits) must be a positive integer.")

    # Validate ls_range
    if (
        not isinstance(ls_range, tuple)
        or len(ls_range) != 2
        or not all(isinstance(val, float) for val in ls_range)
    ):
        raise ValueError("ls_range must be a tuple of two np.number values.")

    # Validate random_seed
    if random_seed is not None and not isinstance(random_seed, int):
        raise ValueError("random_seed must be None or an integer.")

    # Either difference or detrend the set of provided signals
    if method == "difference":
        out_df = _difference_all_signals(df, signal_id, timestamp, value_col)
    else:
        if detrend_type == "lr":
            out_df = _detrend_all_signals_linreg(
                df, signal_id=signal_id, timestamp=timestamp, value_col=value_col
            )
        elif detrend_type == "gp":
            ls_low, ls_high = ls_range
            if random_seed is None:
                ls_vals = np.random.uniform(ls_low, ls_high, size=n_searches)
            else:
                rng = np.random.default_rng(random_seed)
                ls_vals = rng.uniform(ls_low, ls_high, size=n_searches)

            out_df = _detrend_all_signals_gp(
                df,
                ls_vals,
                signal_id=signal_id,
                timestamp=timestamp,
                value_col=value_col,
                n_splits=n_splits,
                eps=eps,
            )

    # Finally, ensure the differenced or detrended signals are statistically stationary
    _, stationary_signals = determine_stationary_signals(
        out_df, alpha, signal_id=signal_id, timestamp=timestamp, value_col=value_col
    )

    stationary_df = out_df.loc[out_df[signal_id].isin(stationary_signals)]
    return stationary_df
