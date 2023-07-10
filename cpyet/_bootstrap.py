import math
import random

import numba as nb
import numpy as np
import pandas as pd

from cpyet.utils import _abs_acorr, _acv_arr, _unif_to_geom, _seed


@nb.njit("i4(f8[:], f8[:], i4, i4)", fastmath=True)
def _compute_mhat(x: np.ndarray, acv: np.ndarray, kn: int, m_max: int) -> int:
    n = x.size

    # Find first collection of kn autocorrelations that are insignificant
    cv = 2 * math.sqrt(math.log10(n) / n)
    abs_acorr = _abs_acorr(acv)
    mhat = m_max

    for i in range(kn, m_max + 1):
        if np.all(abs_acorr[i - kn : i] < cv):
            mhat = i - kn
            break

    return min(2 * max(mhat, 1), m_max)


@nb.njit("f8(f8)", fastmath=True)
def _h(x: float) -> float:
    return min(1.0, 2 * (1 - abs(x)))


@nb.njit("f8(f8[:], f8)", fastmath=True)
def _optimal_block_size(x: np.ndarray, c: float = 2.0) -> float:
    n = x.size
    kn = max(5, int(math.log10(n)))
    m_max = int(math.ceil(math.sqrt(n))) + kn
    acv = _acv_arr(x, m_max + 1)
    b_max = min(3.0 * math.sqrt(n), n / 3.0)

    mhat = _compute_mhat(x, acv, kn, m_max)

    g = 0.0
    sigma = acv[0]

    # By symmetry we only do half the iterations then multiply g and sigma
    # by a factor of two
    for k in range(1, mhat + 1):
        z = k / mhat
        gamma = acv[k]
        lam = _h(z)
        sigma += 2.0 * lam * gamma
        g += 2.0 * lam * k * gamma

    # c = 2.0 is the tuning parameter listed by Andrew Patton; original
    # coder for this approach in the Politis and Romano (2004) paper
    d = c * (sigma) ** 2
    b_star = ((2 * g**2) / d) ** (1 / 3) * n ** (1 / 3)
    return min(b_star, b_max)


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


@nb.njit("f8[:, :](f8[:], i4, f8)")
def _stationary_bootstrap(
    x: np.ndarray, n_boot: int = 1000, c: float = 2.0
) -> np.ndarray:
    """
    Generates `n_boot` stationary bootstrap samples

    Args:
        x (np.ndarray): Time series signal. Shape (n,).
        n_boot (int, optional): Number of bootstrap samples. Default is n_boot = 1000.
        c (float, optional): Tuning parameter for optimal block size selection. Default is c = 2.0.

    Returns:
        np.ndarray: Bootstrapped time series matrix. Shape: (n_boot, n).
    """
    n = x.size
    X = np.zeros(shape=(n_boot, n), dtype=np.float64)

    bstar = _optimal_block_size(x, c)
    bstar = max(bstar, 1.01)  # Account degenerate case where b* < 1
    pstar = 1.0 / bstar  # E[Geom(p)] = 1 / p

    for i in range(n_boot):
        boot_idx = _single_stationary_boot(n, pstar)
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
