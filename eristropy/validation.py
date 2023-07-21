import numpy as np
import pandas as pd


def _validate_dataframe(
    df: pd.DataFrame,
    signal_id: str = "signal_id",
    timestamp: str = "timestamp",
    value_col: str = "value_col",
) -> None:
    required_columns = [signal_id, timestamp, value_col]
    numeric_cols = [timestamp, value_col]

    # Check if the input is a pandas DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    # Check if the DataFrame is empty
    if df.empty:
        raise ValueError("Input DataFrame is empty")

    # Check if the required columns are present in the DataFrame
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Check if the DataFrame contains complex, timestamp or boolean data
    complex_data_mask = df[required_columns].applymap(np.iscomplex).any(axis=None)
    timestamp_data_mask = any(
        pd.api.types.is_datetime64_any_dtype(df[col]) for col in required_columns
    )
    boolean_data_mask = any(
        pd.api.types.is_bool_dtype(df[col]) for col in required_columns
    )

    if complex_data_mask or timestamp_data_mask or boolean_data_mask:
        raise ValueError("Input DataFrame contains complex, timestamp or boolean data")

    # Convert 'timestamp' and 'value_col' columns to numeric
    for column in numeric_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Check if the DataFrame contains NaN or np.inf values
    if (
        df[numeric_cols].isnull().values.any()
        or np.isinf(df[numeric_cols].values).any()
    ):
        raise ValueError(
            "Input DataFrame contains NaN or np.inf values or non-numeric data."
        )

    # Check to make sure there's at least one unique signal ID
    unique_signal_ids = df[signal_id].nunique()
    if unique_signal_ids == 0:
        raise ValueError("No unique signal IDs")
