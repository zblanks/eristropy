import numpy as np
import pandas as pd


def _difference(x: np.ndarray) -> np.ndarray:
    """
    Compute the differenced signal to make a time series statistically stationary.

    Args:
        x (np.ndarray): The input time series signal.

    Returns:
        np.ndarray: The differenced signal.

    Example:
        >>> x = [1, 3, 6, 10, 15]
        >>> _difference(x)
        array([2, 3, 4, 5])
    """
    if len(x) < 2:
        raise ValueError("Input must have at least two elements.")

    if np.isnan(x).any() or np.isinf(x).any():
        raise ValueError("Input contains NaN or np.inf values.")

    return np.diff(x)


def _difference_all_signals(
    df: pd.DataFrame, signal_id: str, timestamp: str, value_col: str
) -> pd.DataFrame:
    """
    Compute the differenced signals for each unique signal ID in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing signal observations.
        params (StationarySignalParams): Dataclass defining relavant parameters
            for constructing stationary signals

    Returns:
        pd.DataFrame: The differenced signals for each unique signal ID.

    Example:
        >>> df = pd.DataFrame({"signal_id": ["abc", "abc", "def", "def"],
                              "timestamp": [1, 2, 1, 2],
                              "value": [2, 3, 5, 7]})
        >>> difference_all_signals(df, 'signal_id', 'timestamp', 'value')
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
                value_col: diff_values,
            }
        )

        # Append the differenced signal DataFrame to the list
        diff_signals.append(diff_df)

    # Concatenate all the differenced signals into a single DataFrame
    diff_df = pd.concat(diff_signals, ignore_index=True)

    return diff_df
