import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


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
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    lr = LinearRegression()
    lr.fit(X, y)
    yhat = lr.predict(X)
    return y - yhat


def _detrend_all_signals_linreg(
    df: pd.DataFrame,
    signal_id: str,
    timestamp: str,
    value_col: str,
) -> pd.DataFrame:
    """
    Compute the detrended signals via linear regression for each unique signal
    ID in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing signal observations.
        signal_id (str): Column name in df containing the signal IDs.
        timestamp (str): Column name in df containing the timestamps.
        value_col (str): Column name in df containing the values.

    Returns:
        pd.DataFrame: The detrended signals for each unique signal ID.

    Example:
        >>> params = StationarySignalParams()
        >>> df = pd.DataFrame({"signal_id": ["abc", "abc", "def", "def"],
                              "timestamp": [1, 2, 1, 2],
                              "value": [2, 3, 5, 7]})
        >>> _detrend_all_signals_linreg(df, params)
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
        X = group[timestamp].values
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
