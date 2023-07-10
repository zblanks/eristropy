import math

import numba as nb
import numpy as np
import pandas as pd

from cpyet._bootstrap import _build_boot_df
from cpyet.utils import _compute_all_standard_error


@nb.njit("f8(f8[:], i4, f8)", fastmath=True)
def _sampen(x: np.ndarray, m: int, r: float) -> float:
    """
    Compute SampEn of inputs time series, `x` given a fixed m and r

    Args:
        x (np.ndarray): Time series signal. Shape is (n,)
        m (int): Embedding dimension
        r (float): Radial distance

    Returns:
        float: SampEn(x; m, r)
    """
    n = x.size
    run = np.zeros(n, dtype=np.int32)
    lastrun = np.zeros(n, dtype=np.int32)
    m += 1
    a = np.zeros(m, dtype=np.float64)
    b = np.zeros(m, dtype=np.float64)

    for i in range(n - 1):
        nj = n - i - 1
        x1 = x[i]

        for jj in range(nj):
            j = jj + i + 1

            if abs(x[j] - x1) < r:
                run[jj] = lastrun[jj] + 1

                # Increment up to the limit of m + 1 (recall we had m += 1 for runs)
                m1 = min(m, run[jj])
                for order in range(m1):
                    a[order] += 1.0

                    # Need like-to-like comparisons with A so have to account
                    # for boundary conditions
                    if j < n - 1:
                        b[order] += 1.0
            else:
                run[jj] = 0

        # Re-set the run counter for future iterations
        for j in range(nj):
            lastrun[j] = run[j]

    if a[-1] == 0.0:
        return np.nan
    else:
        return -math.log(a[m - 1] / b[m - 2])


@nb.njit(fastmath=True)
def _cp_mean_and_sd(x: np.ndarray, mm: int, r: float) -> tuple[float, float]:
    """
    Implements the CP and analytical CP estimation detailed in Lake et al.
    `Sample entropy analysis of neonatal heart rate variability`

    Args:
        x (np.ndarray): Time series signal. Shape is (n,)
        mm (int): Embedding dimension
        r (float): Radial distance

    Returns:
        tuple[float, float]: Estiamted conditional probability (CP) of A / B,
            and estimated analytical CP standard deviation

    Notes:
        The purpose of these estimates is to feed into Lake et al.'s downstream
        selection of r efficiency criteria

        The validity of this approach is predicated on the following statistical
        assumptions:
            1. Since SampEn = -log(CP), via error of propagations, the standard
                error of SampEn can be estimated as s / CP, where s denotes the
                estimated standard deviation of CP
            2. From the paper: "For m small enough and r large enough to ensure
                a sufficient number of matches, SampEn can be assumed to be
                normally distributed and we define the 95% CI of SampEn to be
                -log(CP) +/- 1.96 * (s / CP)"

        Your milage may vary on both of these assumptions in the context of
        CPET signals. However, they were probably reasonbale for longer R-R signals,
        the context in which they were proposed and studied.
    """
    n = x.size
    mm += 1
    MM = 2 * mm

    run = np.zeros(n, dtype=np.int32)
    run1 = np.zeros(n, dtype=np.int32)
    R1 = np.zeros((n, MM), dtype=np.int32)
    R2 = np.zeros((n, MM), dtype=np.int32)
    F = np.zeros((n, MM), dtype=np.int32)
    F1 = np.zeros((n, mm), dtype=np.int32)
    F2 = np.zeros((n, mm), dtype=np.int32)
    K = np.zeros(((mm + 1) * mm), dtype=np.float64)
    A = np.zeros(mm, dtype=np.float64)
    B = np.zeros(mm, dtype=np.float64)
    p = np.zeros(mm, dtype=np.float64)
    v1 = np.zeros(mm, dtype=np.float64)
    v2 = np.zeros(mm, dtype=np.float64)
    s1 = np.zeros(mm, dtype=np.float64)
    n1 = np.zeros(mm, dtype=np.float64)
    n2 = np.zeros(mm, dtype=np.float64)

    for i in range(n - 1):
        nj = n - i - 1
        x1 = x[i]
        for jj in range(nj):
            j = jj + i + 1
            if abs(x[j] - x1) < r:
                run[jj] = run1[jj] + 1
                m1 = mm if mm < run[jj] else run[jj]
                for m in range(m1):
                    A[m] += 1
                    if j < n - 1:
                        B[m] += 1
                    F1[i, m] += 1
                    F[i, m] += 1
                    F[j, m] += 1
            else:
                run[jj] = 0

        for j in range(MM):
            run1[j] = run[j]
            R1[i, j] = run[j]

        if nj > MM - 1:
            for j in range(MM, nj):
                run1[j] = run[j]

    for i in range(1, MM):
        for j in range(i - 1):
            R2[i, j] = R1[i - j - 1, j]
    for i in range(MM, n):
        for j in range(MM):
            R2[i, j] = R1[i - j - 1, j]

    for i in range(n):
        for m in range(mm):
            FF = F[i, m]
            F2[i, m] = FF - F1[i, m]
            K[m * (mm + 1)] += FF * (FF - 1)

    for m in range(mm - 1, 0, -1):
        B[m] = B[m - 1]
    B[0] = n * (n - 1) / 2
    for m in range(mm):
        p[m] = A[m] / B[m]
        v2[m] = p[m] * (1 - p[m]) / B[m]

    dd = 1
    for m in range(mm):
        d2 = m + 1 if m + 1 < mm - 1 else mm - 1
        for d in range(d2 + 1):
            for i1 in range(d + 1, n):
                i2 = i1 - d - 1
                nm1 = F1[i1, m]
                nm3 = F1[i2, m]
                nm2 = F2[i1, m]
                nm4 = F2[i2, m]
                for j in range(dd - 1):
                    if R1[i1, j] >= m + 1:
                        nm1 -= 1
                    if R2[i1, j] >= m + 1:
                        nm4 -= 1
                for j in range(2 * (d + 1)):
                    if R2[i1, j] >= m + 1:
                        nm2 -= 1
                for j in range(2 * d + 1):
                    if R1[i2, j] >= m + 1:
                        nm3 -= 1
                K[d + 1 + (mm + 1) * m] += 2 * (nm1 + nm2) * (nm3 + nm4)

    n1[0] = n * (n - 1) * (n - 2)
    for m in range(mm - 1):
        for j in range(m + 2):
            n1[m + 1] += K[j + (mm + 1) * m]

    for m in range(mm):
        for j in range(m + 1):
            n2[m] += K[j + (mm + 1) * m]

    for m in range(mm):
        v1[m] = v2[m]
        dv = (n2[m] - n1[m] * p[m] * p[m]) / (B[m] * B[m])
        if dv > 0:
            v1[m] += dv
        s1[m] = math.sqrt(v1[m])

    return p[-1], s1[-1]


def compute_all_sampen(
    df: pd.DataFrame,
    m: int,
    r: float,
    signal_id: str = "signal_id",
    value_col: str = "value",
    boot_col: str = None,
) -> pd.DataFrame:
    """
    Computes the SampEn of all unique input signals

    Args:
        df (pd.DataFrame): The input DataFrame containing the signals.
        m (int): Embedding dimension
        r (float): Radial distance
        signal_id (str, optional): The column name for the signal ID. Default is 'signal_id'.
        value_col (str, optional): The column name for the signal values. Default is 'value'.
        boot_col (str, optional): The column name if also computing SampEn of bootstrapped
            time series.

    Returns:
        pd.DataFrame: DataFrame containing the estimated SampEn, given `m` and `r`
        for all unique signals

    Notes:
        The validity of the SampEn is contingent upon the input signals being
        statistically stationary. If one provides non-stationary signals, then
        it is likely the resulting entropy estimate will be incorrect.
    """
    if boot_col is not None:
        cols = [signal_id, boot_col]
    else:
        cols = [signal_id]

    sampen = df.groupby(cols).apply(lambda x: _sampen(x[value_col].values, m, r))
    sampen = sampen.to_frame()
    sampen = sampen.reset_index()

    cols.append("sampen")
    sampen.columns = cols
    return sampen


def find_rstar(
    df: pd.DataFrame,
    m: int,
    rs: np.ndarray = None,
    signal_id: str = "signal_id",
    timestamp: str = "timestamp",
    value_col: str = "value",
    random_seed: int = None,
    n_boot: int = 1000,
    c: float = 2.0,
) -> tuple[float, pd.DataFrame]:
    """
    Finds the best SampEn radial distance by optimizing the Lake et al. SampEn
    efficiency metric

    Args:
        df (pd.DataFrame): The input DataFrame containing the signals.
        m: (int): Embedding dimension
        rs (np.ndarray, optional): Array of radial distances to consider.
        signal_id (str, optional): The column name for the signal ID. Default is 'signal_id'.
        timestamp (str, optional): The column name for the timestamp. Default is 'timestamp'.
        value_col (str, optional): The column name for the signal values. Default is 'value'.
        random_seed (int, optional): Random seed for reproducibility
        n_boot (int, optional): Number of bootstrap iterates for one `signal_id`. Defualt is 1000.
        c (float, optional): Tuning parameter for the block bootstrap block size selection.
            Default is `c = 2.0`.

    Returns:
        tuple[float, pd.DataFrame]: Optimal rstar value with the corresponding
            DataFrame containing the estimates given rstar and m.
    """
    if rs is None:
        sigma = df[value_col].std()
        rs = sigma * np.array([0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])

    boot_df = _build_boot_df(
        df, signal_id, timestamp, value_col, random_seed=random_seed, n_boot=n_boot, c=c
    )

    best_obj_val = np.inf
    rstar = np.nan
    sampen_out = pd.DataFrame()

    for r in rs:
        sampen_df = compute_all_sampen(
            df, m, r=r, signal_id=signal_id, value_col=value_col
        )

        boot_sampen = compute_all_sampen(
            boot_df,
            m,
            r=r,
            signal_id=signal_id,
            value_col=value_col,
            boot_col="boot_num",
        )

        # It's possible there were some NaN bootstrap iterates due to there
        # being no matches -- just drop those rows
        boot_sampen = boot_sampen.dropna()

        se_df = _compute_all_standard_error(boot_sampen, signal_id, var_name="sampen")
        tmp_df = sampen_df.merge(se_df, how="inner", on=signal_id)
        tmp_df["normalized_sampen_se"] = tmp_df["sampen_se"] / tmp_df["sampen"]
        tmp_df["sampen_eff"] = tmp_df[["sampen_se", "normalized_sampen_se"]].max(axis=1)

        obj_val = tmp_df["sampen_eff"].median()
        if obj_val < best_obj_val:
            best_obj_val = obj_val
            rstar = r
            sampen_out = sampen_df.copy()

    return rstar, sampen_out
