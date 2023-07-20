import random

import numba as nb
import numpy as np
import pandas as pd

from eristropy.utils import _unif_to_geom, _seed


@nb.njit("i4[:](i4, f8)")
def _get_idx(n: int, p: float) -> np.array:
    t_start = random.randint(0, n - 1)
    u = random.random()
    b = _unif_to_geom(u, p)

    # Have to wrap around if tstart + b >= n to ensure bootstrap stationarity
    if t_start + b >= n:
        idx1 = np.arange(start=t_start, stop=n, dtype=np.int32)
        idx2 = np.arange(start=0, stop=(b - (n - t_start)), dtype=np.int32)
        idx = np.concatenate((idx1, idx2))
    else:
        idx = np.arange(start=t_start, stop=t_start + b, dtype=np.int32)

    return idx


@nb.njit("i4[:](i4, f8)")
def _single_stationary_boot(n: int, p: float) -> np.ndarray:
    arr = np.zeros((n,), dtype=np.int32)
    s = 0

    while s < n:
        idx = _get_idx(n, p)
        if idx.size > n - s:
            # Truncate to ensure consistent time series length
            idx = idx[: n - s]

        b = idx.size
        arr[s : s + b] = idx
        s += b

    return arr


@nb.njit("f8[:, :](f8[:], f8, i4)")
def _stationary_bootstrap(x: np.ndarray, p: float, n_boot: int = 1000) -> np.ndarray:
    """
    Generates `n_boot` stationary bootstrap samples

    Args:
        x (np.ndarray): Time series signal. Shape (n,).
        p (float): Geometric distribution success probability for stationary bootstrap.
        n_boot (int, optional): Number of bootstrap samples. Default is n_boot = 1000.

    Returns:
        np.ndarray: Bootstrapped time series matrix. Shape: (n_boot, n).
    """
    n = x.size
    X = np.zeros(shape=(n_boot, n), dtype=np.float64)

    for i in range(n_boot):
        boot_idx = _single_stationary_boot(n, p)
        X[i, :] = x[boot_idx]

    return X


def _build_boot_df(
    df: pd.DataFrame,
    signal_id: str = "signal_id",
    timestamp: str = "timestamp",
    value_col: str = "value",
    random_seed: int = None,
    n_boot: int = 1000,
    c: float = 2.0,
) -> pd.DataFrame:
    """
    Constructs DataFrame of all bootstrap samples for all unique `signal_id`

    Args:
        df (pd.DataFrame): The input DataFrame containing the signals.
        signal_id (str, optional): The column name for the signal ID. Default is 'signal_id'.
        timestamp (str, optional): The column name for the timestamp. Default is 'timestamp'.
        value_col (str, optional): The column name for the signal values. Default is 'value'.
        random_seed (int, optional): Random seed for reproducibility
        n_boot (int, optional): Number of bootstrap iterates for one `signal_id`. Defualt is 1000.
        c (float, optional): Tuning parameter for the block bootstrap block size selection.
            Default is `c = 2.0`.

    Returns:
        pd.DataFrame: DataFrame of all bootstrap samples for all unique `signal_id`.

    Notes:
        For this function to be valid, the input signals must be stationary.
        The stationary bootstrap procedure assumes the input signal is stationary.
    """
    out = []
    unique_signals = df[signal_id].unique()

    if random_seed is not None:
        _seed(random_seed)

    for signal in unique_signals:
        x = df.loc[df[signal_id] == signal, value_col].values.astype(np.float64)
        Xboot = _stationary_bootstrap(x, n_boot, c)  # Shape = (n_boot, n)

        n = Xboot.shape[1]
        signal_arr = np.repeat([signal], repeats=(n_boot * n))
        time_arr = np.tile(
            df.loc[df[signal_id] == signal, timestamp].values, reps=n_boot
        )
        boot_count_arr = np.repeat(np.arange(n_boot), n)
        out.append(
            pd.DataFrame(
                {
                    signal_id: signal_arr,
                    timestamp: time_arr,
                    "boot_num": boot_count_arr,
                    value_col: Xboot.flatten(),
                }
            )
        )

    return pd.concat(out, ignore_index=True)
