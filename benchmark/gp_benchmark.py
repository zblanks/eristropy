import sys
import time

import numpy as np
import pandas as pd

from cpyet._gp import _detrend_all_signals_gp_numba, _detrend_all_signals_gp_sklearn


def make_benchmark_dataframe(t: int, n: int) -> pd.DataFrame:
    """
    Constructs the benchmark DataFrame for evaluating the Scikit-Learn versus
    CPyET implementation of GP de-trending.

    Args:
        t (int): Signal length
        n (int): Number of unique signals

    Returns:
        pd.DataFrame: Benchmark DataFrame

    Notes:
        This approach constructs a linear CPET signal trend, roughly
        approximating the VO2-time relationship
    """
    signal_ids = np.repeat(np.arange(n), t)
    T = np.tile(np.arange(t).reshape(-1, 1), (1, n))

    # For this example, we assume a general upward trend with a VO2 signal
    # that terminates between [3/N, 5/N] where N denotes the `signal_len`
    # v^(i) := (b * t) + noise; noise ~ N(0, 0.1)
    # Thus we will generate uniform values for b and e to generate the
    # benchmark signals
    rng = np.random.default_rng(17)
    bs = rng.uniform(low=3 / t, high=5 / t, size=(n,))
    noise = rng.normal(loc=0.0, scale=0.1, size=(t, n))
    values = (bs * T) + noise

    df = pd.DataFrame(
        {
            "signal_id": signal_ids,
            "timestamp": np.transpose(T).flatten(),
            "value": values.T.flatten(),
        }
    )

    return df


if __name__ == "__main__":
    args = sys.argv
    N = int(sys.argv[1])
    T = int(sys.argv[2])
    method = sys.argv[3]

    rng = np.random.default_rng(17)
    ls_vals = rng.uniform(10.0, 100.0, size=(10,))

    # Throw out first Numba run (w/ compilation)
    _detrend_all_signals_gp_numba(make_benchmark_dataframe(t=10, n=2), ls_vals)

    df = make_benchmark_dataframe(T, N)

    if method == "numba":
        start_time = time.perf_counter()
        _detrend_all_signals_gp_numba(df, ls_vals)
        end_time = time.perf_counter() - start_time
    elif method == "sklearn":
        start_time = time.perf_counter()
        _detrend_all_signals_gp_sklearn(df, random_seed=17)
        end_time = time.perf_counter() - start_time

    print(f"{N},{T},{method},{end_time}")
