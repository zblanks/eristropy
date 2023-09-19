import math
import warnings

import numba as nb
import numpy as np
import optuna
from optuna.samplers import TPESampler
import pandas as pd

from eristropy.bootstrap import _stationary_bootstrap
from eristropy.validation import _validate_dataframe
from eristropy.utils import _mean, _mean_squared_error, _sampen, _seed, _standard_error


class SampEnSettingWarning(UserWarning):
    """Warns regarding SampEn setting boundary conditions"""

    pass


class OptimizationFailureWarning(UserWarning):
    """Warning for when an optimization routine fails."""

    pass


class SampleEntropy:
    """
    Class for computing the Sample Entropy of multiple signals and finding the optimal
    SampEn parameters using a regularized MSE objective function and the
    Optuna Bayesian optimization framework with the Tree-based Parzen Estimator (TPE)
    surrogate function.

    The main methods are `find_optimal_sampen_params` to perform the optimization and
    `compute_all_sampen` to compute the SampEn of all signals with the optimized or provided
    parameters.

    Attributes:
        df (pd.DataFrame): DataFrame containing the signals. The DataFrame must
            contain columns for the signal_id, timestamp, and signal_value at each step.
        signal_id (str): The column name for signal id in the DataFrame. Default is "signal_id".
        timestamp (str): The column name for timestamps in the DataFrame. Default is "timestamp".
        value_col (str): The column name for signal values in the DataFrame. Default is "value".
        objective (str): Valid options are "mse" and "sampen_eff". Default is mse.
        n_boot (int): The number of bootstrap samples to use in the estimation. Default is 100.
        n_trials (int): The number of trials for the optimization. Default is 100.
        random_seed (int): The random seed for reproducibility. Default is None (no seeding).
        r_range (tuple): The range of r values for the optimization. Default is (0.10, 0.50).
        m_range (tuple): The range of m values for the optimization. Default is (1, 3).
        p_range (tuple): The range of p values for the stationary bootstrap. Default is (0.01, 0.99).
        lam (float): The trade-off parameter between the r-based penalization. Default is 0.33.
        r (float): The user-provided value for r. Default is None (to be optimized).
        m (int): The user-provided value for m. Default is None (to be optimized).
        p (float): The user-provided value for p. Default is None (to be optimized).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        signal_id: str = "signal_id",
        timestamp: str = "timestamp",
        value_col: str = "value",
        objective: str = "mse",
        n_boot: int = 100,
        n_trials: int = 100,
        random_seed: int = None,
        r_range: tuple[float, float] = (0.10, 0.50),
        m_range: tuple[int, int] = (1, 3),
        p_range: tuple[float, float] = (0.01, 0.99),
        lam: float = 0.33,
        r: float = None,
        m: int = None,
        p: float = None,
    ) -> None:
        self.df = df
        self.signal_id = signal_id
        self.timestamp = timestamp
        self.value_col = value_col
        self.objective = objective
        self.n_boot = n_boot
        self.n_trials = n_trials
        self.random_seed = random_seed
        self.r_range = r_range
        self.m_range = m_range
        self.p_range = p_range
        self.lam = lam
        self.r = r
        self.m = m
        self.p = p
        self.study_ = None  # Storage for optimization results

        _validate_dataframe(self.df, self.signal_id, self.timestamp, self.value_col)

        for arg in [self.signal_id, self.timestamp, self.value_col]:
            if not isinstance(arg, str):
                raise ValueError(f"{arg} should be a string. Received {type(arg)}")

        self._check_ranges()
        self._check_fixed_values()
        self._check_positive_integer(self.n_boot, "n_boot")
        self._check_positive_integer(self.n_trials, "n_trials")
        if self.random_seed is not None:
            if not isinstance(self.random_seed, int) or self.random_seed < 0:
                raise ValueError("random_seed must be a non-zero integer")

        min_signal_length = df.groupby(self.signal_id).size().min()
        if self.m_range[1] > min_signal_length:
            raise ValueError(
                "The upper limit of m_range cannot exceed the length of the shortest "
                "time series in the DataFrame."
            )

        # Initialize optimization variables (probably will be None)
        self.m_star_ = self.m
        self.r_star_ = self.r
        self.p_star_ = self.p

        # Initialize constants for downstream computation
        self.unique_signals_ = self.df[self.signal_id].unique()
        self.signal_groups_ = self.df.groupby(self.signal_id)

    def _check_string_attributes(self, *args):
        for arg in args:
            if not isinstance(getattr(self, arg), str):
                raise ValueError(
                    f"{arg} should be a string. Received {type(getattr(self, arg))}"
                )

    def _check_ranges(self):
        for arg_name in ["r_range", "m_range", "p_range"]:
            arg_value = getattr(self, arg_name)
            if arg_value is not None:
                if not isinstance(arg_value, tuple) or len(arg_value) != 2:
                    raise ValueError(f"{arg_name} must be a tuple of two elements.")
                if arg_value[1] <= arg_value[0]:
                    raise ValueError(
                        f"Second element of {arg_name} must be greater than the first."
                    )

                if arg_name == "p_range":
                    if not (0 < arg_value[0] < arg_value[1] < 1):
                        raise ValueError(
                            "In p_range, the first element must be > 0 and the second element < 1."
                        )
                elif arg_name == "r_range":
                    if arg_value[0] <= 0:
                        raise ValueError("In r_range, the first element must be > 0.")
                elif arg_name == "m_range":
                    if not (
                        isinstance(arg_value[0], int) and isinstance(arg_value[1], int)
                    ):
                        raise ValueError("In m_range, both elements must be integers.")

    def _check_fixed_values(self):
        for arg_name in ["r", "m", "p", "lam"]:
            arg_value = getattr(self, arg_name)
            if arg_value is not None:
                if arg_name == "m":
                    if not isinstance(arg_value, int) or arg_value <= 0:
                        raise ValueError("m must be an integer > 0.")
                elif arg_name == "r":
                    if arg_value <= 0:
                        raise ValueError(f"{arg_name} must be > 0.")
                elif arg_name == "p":
                    if not isinstance(arg_value, float) or not (0 < arg_value < 1):
                        raise ValueError(
                            "p must be a float between 0 and 1 (exclusive)."
                        )
                elif arg_name == "lam":
                    if arg_value < 0:
                        raise ValueError("lam must be >= 0")

        if self.r is not None:
            if not (self.r_range[0] < self.r < self.r_range[1]):
                warnings.warn(
                    "Provided r value is at the boundary of the r_range. "
                    "This may affect the reliability of SampEn estimation. "
                    "Consider setting r to a more moderate value within the r_range.",
                    SampEnSettingWarning,
                )

        if self.p is not None:
            if not (self.p_range[0] < self.p < self.p_range[1]):
                warnings.warn(
                    "Provided p value is at the boundary of the p_range. "
                    "This may affect the reliability of the bootstrap process. "
                    "Consider setting p to a more moderate value within the p_range.",
                    SampEnSettingWarning,
                )

    @staticmethod
    def _check_positive_integer(value, name):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer. Received {value}")

        warning_thresholds = {"n_boot": 50, "n_trials": 10}
        if name in warning_thresholds.keys() and value < warning_thresholds[name]:
            warnings.warn(
                f"{name} is set to a low value. This may affect the reliability of the SampEn "
                "estimation and optimization. Consider setting it to a higher value, "
                f"preferably at least {warning_thresholds[name]}.",
                SampEnSettingWarning,
            )

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

    @staticmethod
    @nb.njit("f8(f8[:], i4, f8, f8, i4)")
    def _sampen_se(x: np.ndarray, m: int, r: float, p: float, n_boot: int) -> float:
        """
        Computes the SE(SampEn(x, m, r)) with the optimal (m, r) values

        Returns:
            float: Bootstrap estimate of SE(SampEn)
        """

        sampen_hat = np.zeros(n_boot, dtype=np.float64)
        X = _stationary_bootstrap(x, p, n_boot)

        for i in range(n_boot):
            sampen_hat[i] = _sampen(X[i, :], m, r)

        return _standard_error(sampen_hat)

    def _mse_objective(self, trial: optuna.trial.Trial) -> float:
        """
        Defines regularized MSE objective for optuna framework
        """
        m = self.m_star_
        r = self.r_star_
        p = self.p_star_

        if m is None:
            m = trial.suggest_int("m", *self.m_range)
        if r is None:
            r = trial.suggest_float("r", *self.r_range)
        if p is None:
            p = trial.suggest_float("p", *self.p_range)

        mse = np.zeros(self.unique_signals_.size, dtype=np.float64)
        for i, (_, group) in enumerate(self.signal_groups_):
            x = group[self.value_col].values.astype(np.float64)
            mse[i] = self._bootstrap_mse(x, m, r, p, self.n_boot)

        obj = _mean(mse) + self.lam * math.sqrt(r)
        return obj

    def _sampen_eff_objective(self, trial: optuna.trial.Trial) -> float:
        """
        Defines bootstrapped SampEn efficiency objective for optuna framework
        """
        m = self.m_star_
        r = self.r_star_
        p = self.p_star_

        if m is None:
            m = trial.suggest_int("m", *self.m_range)
        if r is None:
            r = trial.suggest_float("r", *self.r_range)
        if p is None:
            p = trial.suggest_float("p", *self.p_range)

        objs = np.zeros(self.unique_signals_.size, dtype=np.float64)
        for i, (_, group) in enumerate(self.signal_groups_):
            x = group[self.value_col].values.astype(np.float64)
            se = self._sampen_se(x, m, r, p, self.n_boot)
            sampen = _sampen(x, m, r)
            objs[i] = max(se, se / sampen)

        obj = np.median(objs)
        return obj

    def _set_objective(self) -> None:
        """
        Sets the objective function for the optimization routine.
        """

        if self.objective == "mse":
            self._objective = self._mse_objective
        elif self.objective == "sampen_eff":
            self._objective = self._sampen_eff_objective

    def find_optimal_sampen_params(self) -> None:
        """
        Finds the optimal (m, r) SampEn parameters for the input signal set
        """

        if self.random_seed is not None:
            _seed(self.random_seed)

        self._set_objective()

        sampler = TPESampler(seed=self.random_seed)
        self.study_ = optuna.create_study(sampler=sampler)

        # NaN values may happen with SampEn values, this isn't something to worry
        # about though
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        try:
            self.study_.optimize(
                self._objective, n_trials=self.n_trials, show_progress_bar=True
            )

            best_params = self.study_.best_params
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

    def compute_all_sampen(
        self, optimize: bool = False, estimate_uncertainty: bool = False
    ) -> pd.DataFrame:
        """
        Computes the SampEn of the input signal set given either the provided
        or optimized values of (m, r)

        Parameters:
            optimize (bool, optional): If True, optimize the SampEn parameters
                before computing the SampEn for all signals. Defaults to False.
            estimate_uncertainty(bool, optional): If True, estiamtes the SE(SampEn)
                for the given or optimized (m, r) values. Defaults to False.

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

        if estimate_uncertainty and self.p_star_ is None:
            raise ValueError(
                "Cannot estimate uncertainty without stationary bootstrap probability value. "
                "Either find the optimal value or provide one before running this function"
            )

        n = self.unique_signals_.size
        out = pd.DataFrame(
            {
                self.signal_id: self.unique_signals_,
                "sampen": np.zeros(n, dtype=np.float64),
            }
        )

        if estimate_uncertainty:
            out["se_sampen"] = np.zeros(n, dtype=np.float64)

        for signal in self.unique_signals_:
            x = self.signal_groups_.get_group(signal)[self.value_col]
            x = x.values.astype(np.float64)
            sampen = _sampen(x, self.m_star_, self.r_star_)
            out.loc[out[self.signal_id] == signal, "sampen"] = sampen

            if estimate_uncertainty:
                se = self._sampen_se(
                    x, self.m_star_, self.r_star_, self.p_star_, self.n_boot
                )
                out.loc[out[self.signal_id] == signal, "se_sampen"] = se

        return out

    def get_optimization_results(
        self, attrs=("number", "value", "params")
    ) -> pd.DataFrame:
        """
        Return a DataFrame of the optimization results.

        Args:
            attrs (tuple, optional): Attributes of the optuna trials to include in the dataframe.
                By default it includes the trial number, value of the objective function and
                parameters used. Refer to optuna documentation for other options.

        Returns:
            pd.DataFrame: DataFrame of the optimization trials.
        """
        if self.study_ is not None:
            return self.study_.trials_dataframe(attrs=attrs)
        else:
            raise ValueError(
                "No optimization results available. The method `find_optimal_sampen_params` "
                "needs to be called before accessing optimization results."
            )
