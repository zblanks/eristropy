import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.multitest import multipletests

from ._difference import _difference_all_signals
from ._gp import _detrend_all_signals_gp_numba, _detrend_all_signals_gp_sklearn
from ._linreg import _detrend_all_signals_linreg


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
        to compute the p-values for each unique signal ID in the DataFrame.
        If a signal is not long enough to estimate the p-value, a p-value of
        1.0 is assigned.

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
            if the DataFrame is empty,
            if the significance level is invalid,
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
        >>> print(stationary_fraction)
        0.5
        >>> print(stationary_signals)
        array(["def"])

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
    gp_implementation: str = "sklearn",
) -> pd.DataFrame:
    """
    Create stationary signals using the specified method.

    Args:
        df (pd.DataFrame): The input DataFrame containing the signals.
        method (str): The method to make the signals stationary. Valid options are 'difference' and 'detrend'.
        detrend_type (str): The type of de-trending. Required when method is 'detrend'.
            Valid options are 'lr' and 'gp'.
        signal_id (str): The column name for the signal ID. Default is 'signal_id'.
        timestamp (str): The column name for the timestamp. Default is 'timestamp'.
        value_col (str): The column name for the signal values. Default is 'value'.
        alpha (float): The significance level for the stationarity test.
            Must be in the range (0, 1). Default is 0.05.
        random_seed (int): The random seed for generating ls_vals.
            Must be an integer. Default is None.
        ls_range (tuple[float, float]): The range of ls values for GP detrending.
            The lower bound must be greater than 0. Default is (1.0, 100.0).
        n_searches (int): The number of ls values to consider for GP detrending.
            Must be a positive integer. Default is 10.
        n_splits (int): The number of cross-validation splits.
            Must be a positive integer. Default is 5.
        eps (float, optional): Small value added to the kernel matrix for numerical stability (default: 1e-6).
        gp_implementation (str, optional): Which GP de-trending to use when method is 'detrend' and detrend_type is 'gp'.
            Valid options are numba and sklearn.


    Returns:
        pd.DataFrame: The DataFrame with stationary signals.

    Raises:
        TypeError: If the input is not a pandas DataFrame.
        ValueError: If the input DataFrame is empty,
            if the input columns are missing,
            if the method or detrend_type values are invalid,
            if the alpha value is not within the valid range,
            if the ls_range lower bound is not greater than 0,
            if n_searches or n_splits are not positive integers.

    Notes:
        We have implemented an RBF-based GP detrending in Numba using a
        standard Cholesky factorization solution. This method, on average,
        is faster than the Scikit-Learn implementation. The benefit will
        become more noticeable as the number of unique signals, $N$, increases.
        However, this implementation is less numerically stable and less well-tested
        than the Scikit-Learn version. You can control which version you use
        via the argument `gp_implementation`.

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

    # Validate the gp_implementation argument
    if method == "detrend" and detrend_type == "gp":
        if gp_implementation not in ["sklearn", "numba"]:
            raise ValueError(
                "Invalid value for 'gp_implementation'. Must be either 'sklearn' or 'numba'."
            )

    # Either difference or detrend the set of provided signals
    if method == "difference":
        out_df = _difference_all_signals(df, signal_id, timestamp, value_col)
    else:
        if detrend_type == "lr":
            out_df = _detrend_all_signals_linreg(
                df, signal_id=signal_id, timestamp=timestamp, value_col=value_col
            )
        elif detrend_type == "gp":
            if gp_implementation == "numba":
                ls_low, ls_high = ls_range
                if random_seed is None:
                    ls_vals = np.random.uniform(ls_low, ls_high, size=n_searches)
                else:
                    rng = np.random.default_rng(random_seed)
                    ls_vals = rng.uniform(ls_low, ls_high, size=n_searches)

                    out_df = _detrend_all_signals_gp_numba(
                        df,
                        ls_vals,
                        signal_id=signal_id,
                        timestamp=timestamp,
                        value_col=value_col,
                        n_splits=n_splits,
                        eps=eps,
                    )
            elif gp_implementation == "sklearn":
                out_df = _detrend_all_signals_gp_sklearn(df, random_seed)

    # Finally, ensure the differenced or detrended signals are statistically stationary
    _, stationary_signals = determine_stationary_signals(
        out_df, alpha, signal_id=signal_id, timestamp=timestamp, value_col=value_col
    )

    stationary_df = out_df.loc[out_df[signal_id].isin(stationary_signals)]
    return stationary_df
