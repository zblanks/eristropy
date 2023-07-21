import math
import warnings

import numba as nb
import numpy as np
import optuna
from optuna.samplers import TPESampler
import pandas as pd

from eristropy.bootstrap import _stationary_bootstrap
from eristropy.dataclasses import SampEnSettings
from eristropy.validation import _validate_dataframe
from eristropy.utils import _mean, _mean_squared_error, _sampen, _seed


class OptimizationFailureWarning(UserWarning):
    """Warning for when an optimization routine fails."""

    pass


class SampleEntropy:
    """
    Class for computing the Sample Entropy of multiple signals and finding the optimal
    SampEn parameters using a regularized MSE objective function and the
    Optuna Bayesian optimization framework with the Tree-based Parzen Estimator (TPE)
    surrogate function.

    The class is initialized with a DataFrame containing the signals and optionally an instance
    of the SampEnParams dataclass to specify parameters. If no SampEnParams instance is provided,
    default parameters are used.

    The main methods are `find_optimal_sampen_params` to perform the optimization and
    `compute_all_sampen` to compute the SampEn of all signals with the optimized or provided
    parameters.

    Attributes:
        df (pd.DataFrame): DataFrame containing the signals. The DataFrame must
            contain columns for the signal_id, timestamp, and signal_value at each step.
        params (SampEnParams): An instance of the SampEnParams dataclass specifying the
            parameters for the SampEn computation and optimization.
    """

    def __init__(self, df: pd.DataFrame, params: SampEnSettings = None) -> None:
        self.df = df

        if params is None:
            self.params = SampEnSettings()
        elif isinstance(params, SampEnSettings):
            self.params = params
        else:
            raise TypeError("params must be an instance of SampEnSettings.")

        _validate_dataframe(
            self.df, self.params.signal_id, self.params.timestamp, self.params.value_col
        )

        min_signal_length = df.groupby(self.params.signal_id).size().min()
        if self.params.m_range[1] > min_signal_length:
            raise ValueError(
                "The upper limit of m_range cannot exceed the length of the shortest "
                "time series in the DataFrame."
            )

        # Initialize optimization variables (probably will be None)
        self.m_star_ = self.params.m
        self.r_star_ = self.params.r
        self.p_star_ = self.params.p

        # Initialize constants for downstream computation
        self.unique_signals_ = self.df[self.params.signal_id].unique()
        self.signal_groups_ = self.df.groupby(self.params.signal_id)

    @staticmethod
    @nb.njit("f8(f8[:], i4, f8, f8, i4)")
    def _bootstrap_mse(x: np.ndarray, m: int, r: float, p: float, n_boot: int) -> float:
        """
        Computes the estimated SampEn MSE for a signal signal, `x`.

        Args:
            x (np.ndarray): Time series signal. Shape is (n,)
            m (int): Embedding dimension
            r (float): Similiarity distance
            p (float): Geometric distribution success probability for stationary bootstrap.
            n_boot (int): Number of bootstrap samples

        Returns:
            float: Estimated MSE
        """

        shat = np.zeros(n_boot, dtype=np.float64)
        X = _stationary_bootstrap(x, p, n_boot)

        for i in range(n_boot):
            shat[i] = _sampen(X[i, :], m, r)

        strue = _sampen(x, m, r)
        strue = np.full(n_boot, strue, dtype=np.float64)
        return _mean_squared_error(strue, shat)

    def _objective(self, trial: optuna.trial.Trial) -> float:
        """
        Defines regularized MSE objective for optuna framework
        """
        m = self.m_star_
        r = self.r_star_
        p = self.p_star_

        if m is None:
            m = trial.suggest_int("m", *self.params.m_range)
        if r is None:
            r = trial.suggest_float("r", *self.params.r_range)
        if p is None:
            p = trial.suggest_float("p", *self.params.p_range)

        mse = np.zeros(self.unique_signals_.size, dtype=np.float64)
        for i, (_, group) in enumerate(self.signal_groups_):
            x = group[self.params.value_col].values.astype(np.float64)
            mse[i] = self._bootstrap_mse(x, m, r, p, self.params.n_boot)

        obj = _mean(mse) + self.params.lam1 * math.sqrt(r)
        return obj

    def find_optimal_sampen_params(self) -> None:
        """
        Finds the optimal (m, r) SampEn parameters for the input signal set
        """

        if self.params.random_seed is not None:
            _seed(self.params.random_seed)

        sampler = TPESampler(seed=self.params.random_seed)
        study = optuna.create_study(sampler=sampler)
        try:
            study.optimize(
                self._objective, n_trials=self.params.n_trials, show_progress_bar=True
            )

            best_params = study.best_params
            self.m_star_ = best_params.get("m", self.m_star_)
            self.r_star_ = best_params.get("r", self.r_star_)
            self.p_star_ = best_params.get("p", self.p_star_)
        except ValueError as e:
            if "No trials are completed yet." in str(e):  # Optuna optimization failed
                self.m_star_ = np.nan
                self.r_star_ = np.nan
                self.p_star_ = np.nan
                warnings.warn(
                    "Optimization failed most likely due to one of the following causes:\n"
                    "1. The r_range is too stringent.\n"
                    "2. The signals are too short to estimate the SampEn.\n"
                    "Consider expanding the r_range or using our PermEn class which better handles short signals.",
                    OptimizationFailureWarning,
                )

    def compute_all_sampen(self, optimize: bool = False) -> pd.DataFrame:
        """
        Computes the SampEn of the input signal set given either the provided
        or optimized values of (m, r)

        Parameters:
            optimize (bool, optional): If True, optimize the SampEn parameters
                before computing the SampEn for all signals. Defaults to False.

        Returns:
            pd.DataFrame: SampEn estimates given (m, r) for all signals in the data
        """
        if optimize:
            self.find_optimal_sampen_params()

        if any([val is None or np.isnan(val) for val in [self.m_star_, self.r_star_]]):
            raise ValueError(
                "Invalid m or r values.\n"
                "Consider finding the optimal parameters using the \n"
                "`optimize=True` flag first or check if the provided parameters\n"
                "to SampleEntropy are valid."
            )

        out = pd.DataFrame(
            {
                self.params.signal_id: self.unique_signals_,
                "sampen": np.zeros(self.unique_signals_.size, dtype=np.float64),
            }
        )

        for signal in self.unique_signals_:
            x = self.signal_groups_.get_group(signal)[
                self.params.value_col
            ].values.astype(np.float64)

            sampen = _sampen(x, self.m_star_, self.r_star_)
            out.loc[out[self.params.signal_id] == signal, "sampen"] = sampen

        return out
