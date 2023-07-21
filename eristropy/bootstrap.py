import random

import numba as nb
import numpy as np

from eristropy.utils import _unif_to_geom


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
def _stationary_bootstrap(x: np.ndarray, p: float, n_boot: int = 100) -> np.ndarray:
    """
    Generates `n_boot` stationary bootstrap samples

    Args:
        x (np.ndarray): Time series signal. Shape (n,).
        p (float): Geometric distribution success probability for stationary bootstrap.
        n_boot (int, optional): Number of bootstrap samples. Default is n_boot = 100.

    Returns:
        np.ndarray: Bootstrapped time series matrix. Shape: (n_boot, n).
    """
    n = x.size
    X = np.zeros(shape=(n_boot, n), dtype=np.float64)

    for i in range(n_boot):
        boot_idx = _single_stationary_boot(n, p)
        X[i, :] = x[boot_idx]

    return X
