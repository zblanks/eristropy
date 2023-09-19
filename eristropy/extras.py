import math

import numba as nb
import numpy as np
import pandas as pd

from eristropy.bootstrap import _stationary_bootstrap
from eristropy.utils import _sampen, _seed, _standard_error


class SampEnEfficiency:
    def __init__(
        self,
        df: pd.DataFrame,
        m: int,
        r_range: tuple[float, float],
        signal_id: str = "signal_id",
        value_col: str = "value",
        r_step_size: float = 0.05,
        interpolation_step_size: float = 0.01,
        p: float = 0.5,
        n_boot: int = 100,
        objective: str = "counting",
        random_seed: int = None,
    ) -> None:
        self.df = df
        self.m = m
        self.signal_id = signal_id
        self.value_col = value_col
        self.p = p
        self.n_boot = n_boot
        self.objective = objective
        self.random_seed = random_seed

        r_low = r_range[0]
        r_high = r_range[1]
        self.rs = np.arange(r_low, r_high + interpolation_step_size, r_step_size)

        self._unique_signals = self.df[self.signal_id].unique()
        self._pts = np.arange(
            r_low, r_high + interpolation_step_size, interpolation_step_size
        )

        # Define empty placeholder for optimal r-value and objective function
        self.r_star_ = None
        self.obj_ = None

        if self.random_seed is not None:
            _seed(self.random_seed)

    @staticmethod
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
            short signals. However, they were probably reasonbale for longer R-R signals,
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

    def _counting_obj(self, x: np.ndarray, r: float) -> float:
        try:
            cp, s_cp = self._cp_mean_and_sd(x, self.m, r)
        except ZeroDivisionError:
            return np.nan

        a = s_cp / cp

        if cp == 1:
            sampen = 0.0
            b = 0.0
        else:
            sampen = -math.log(cp)
            b = a * (1.0 / sampen)

        return max(a, b)

    def _bootstrap_obj(self, x: np.ndarray, r: float) -> float:
        X = _stationary_bootstrap(x, self.p, self.n_boot)
        sampens = np.zeros(self.n_boot, dtype=np.float64)

        for i in range(self.n_boot):
            sampens[i] = _sampen(X[i, :], self.m, r)

        se = _standard_error(sampens)
        sampen = _sampen(x, self.m, r)
        return max(se, se / sampen)

    def _sampen_effiency(self, x: np.ndarray, r: float) -> float:
        if self.objective == "counting":
            return self._counting_obj(x, r)
        elif self.objective == "bootstrap":
            return self._bootstrap_obj(x, r)

    def _compute_objective(self) -> np.ndarray:
        M = self._unique_signals.size
        n = self.rs.size
        Y = np.zeros((M, n), dtype=np.float64)

        for i, signal in enumerate(self._unique_signals):
            for j in range(n):
                x = self.df.loc[self.df[self.signal_id] == signal, self.value_col]
                x = x.values.astype(np.float64)
                Y[i, j] = self._sampen_effiency(x, self.rs[j])

        return np.median(Y, axis=0)

    def _interpolate_objective(self) -> np.ndarray:
        objs = self._compute_objective()
        return np.interp(self._pts, self.rs, objs)

    def _sampen_and_se(self, x: np.ndarray, r: float) -> tuple[float, float]:
        cp, s_cp = self._cp_mean_and_sd(x, self.m, r)
        sampen = -math.log(cp)
        se_sampen = s_cp / cp
        return sampen, se_sampen

    def _get_rstar(self) -> None:
        objs = self._interpolate_objective()
        self.r_star_ = self._pts[np.nanargmin(objs)]
        self.obj_ = np.nanmin(objs)

    def compute_all_sampen(self) -> pd.DataFrame:
        self._get_rstar()

        n = self._unique_signals.size
        out_df = pd.DataFrame(
            {
                self.signal_id: self._unique_signals,
                "sampen": np.zeros(n, dtype=np.float64),
                "se_sampen": np.zeros(n, dtype=np.float64),
            }
        )

        for signal in self._unique_signals:
            x = self.df.loc[self.df[self.signal_id] == signal, self.value_col]
            x = x.values.astype(np.float64)
            sampen, se_sampen = self._sampen_and_se(x, self.r_star_)
            out_df.loc[out_df[self.signal_id] == signal, "sampen"] = sampen
            out_df.loc[out_df[self.signal_id] == signal, "se_sampen"] = se_sampen

        return out_df
