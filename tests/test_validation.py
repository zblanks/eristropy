import numpy as np
import pandas as pd
import pytest

from eristropy.validation import _validate_dataframe


def test_empty_dataframe():
    # Empty DataFrame
    with pytest.raises(ValueError):
        _validate_dataframe(pd.DataFrame(), "signal_id", "timestamp", "value")


def test_missing_required_columns():
    # DataFrame with missing required columns
    with pytest.raises(ValueError):
        _validate_dataframe(
            pd.DataFrame({"signal_id": [1, 2, 3]}), "signal_id", "timestamp", "value"
        )


def test_non_numeric_values():
    # DataFrame with non-numeric values
    with pytest.raises(ValueError):
        _validate_dataframe(
            pd.DataFrame(
                {
                    "signal_id": [1, 2, 3],
                    "timestamp": ["a", "b", "c"],
                    "value": [4, 5, 6],
                }
            ),
            "signal_id",
            "timestamp",
            "value",
        )


def test_string_values():
    # DataFrame with string values
    with pytest.raises(ValueError):
        _validate_dataframe(
            pd.DataFrame(
                {
                    "signal_id": ["1", "2", "3"],
                    "timestamp": ["a", "b", "c"],
                    "value": ["4", "5", "6"],
                }
            ),
            "signal_id",
            "timestamp",
            "value",
        )


def test_complex_values():
    # DataFrame with complex values
    with pytest.raises(ValueError):
        _validate_dataframe(
            pd.DataFrame(
                {
                    "signal_id": [1 + 1j, 2 + 2j, 3 + 3j],
                    "timestamp": [1j, 2j, 3j],
                    "value": [4j, 5j, 6j],
                }
            ),
            "signal_id",
            "timestamp",
            "value",
        )


def test_mixed_numeric_types():
    # DataFrame with mixed numeric types
    df_valid = pd.DataFrame(
        {
            "signal_id": [1, 2, 3],
            "timestamp": [1.0, 2.0, 3.0],
            "value": [4.5, 5.5, 6.5],
        }
    )
    _validate_dataframe(df_valid, "signal_id", "timestamp", "value")  # This should pass


def test_nan_inf_values():
    with pytest.raises(ValueError):
        _validate_dataframe(
            pd.DataFrame(
                {
                    "signal_id": [1, 2, 3],
                    "timestamp": [np.nan, 2.0, 3.0],
                    "value": [4.5, np.inf, 6.5],
                }
            ),
            "signal_id",
            "timestamp",
            "value",
        )


def test_inappropriate_data_type_columns():
    # DataFrame with inappropriate data type columns
    with pytest.raises(ValueError):
        _validate_dataframe(
            pd.DataFrame(
                {
                    "signal_id": [1, 2, 3],
                    "timestamp": [
                        pd.Timestamp("2023-01-01"),
                        pd.Timestamp("2023-01-02"),
                        pd.Timestamp("2023-01-03"),
                    ],
                    "value": [True, False, True],
                }
            ),
            "signal_id",
            "timestamp",
            "value",
        )
