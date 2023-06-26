from typing import TypeVar, Union, List, Tuple

import numpy as np

T = TypeVar("T")
ArrayLike = Union[List[T], Tuple[T, ...], np.ndarray]


def check_arraylike(x: ArrayLike):
    if not isinstance(x, (list, tuple, np.ndarray)):
        raise TypeError("Input must be a list, tuple, or NumPy array")


def difference(x: ArrayLike) -> np.ndarray:
    """
    Compute the differenced signal to make a time series (ideally) statistically stationary.

    Parameters:
        x: The input time series signal.

    Returns:
        The differenced signal.

    Example:
        >>> x = [1, 3, 6, 10, 15]
        >>> difference(x)
        array([2, 3, 4, 5])
    """
    check_arraylike(x)

    if len(x) < 2:
        raise ValueError("Input must have at least two elements.")

    if np.isnan(x).any() or np.isinf(x).any():
        raise ValueError("Input contains NaN or np.inf values.")

    return np.diff(x)
