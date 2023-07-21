from dataclasses import dataclass
import warnings

import numpy as np


class SampEnSettingWarning(UserWarning):
    """Warns regarding SampEn setting boundary conditions"""

    pass


@dataclass
class StationarySignalParams:
    """
    Parameters for generating stationary signals.

    Attributes:
        method (str): The method to use to make signals stationary. Choices are
            'difference' and 'detrend'.
        detrend_type (str): The type of detrending to use if method is 'detrend'.
            Choices are 'gp' and 'lr'.
        signal_id (str): The name of the column in the input DataFrame that contains the signal IDs.
        timestamp (str): The name of the column in the input DataFrame that contains the timestamps.
        value_col (str): The name of the column in the input DataFrame that contains the values.
        alpha (float): The significance level for the Augmented Dickey-Fuller test.
        random_seed (int): The seed for the random number generator.
        ls_range (tuple[float, float]): The range of length-scales to consider for
            the Gaussian process.
        ls_vals (np.ndarray): List of considered length scale values
        n_searches (int): The number of searches to perform for the Gaussian process hyperparameters.
        n_splits (int): The number of splits for the cross-validation.
        eps (float): The tolerance for the difference.
        gp_implementation (str): The implementation to use for the Gaussian process.
            Choices are 'sklearn' and 'numba'.
        sklearn_scoring (str): Method to score the GP estimator when calling the
            `sklearn` implementation. See Scikit-learn documentation for additional
            options under RandomizedSearchCV `scoring`
    """

    method: str = "detrend"
    detrend_type: str = "gp"
    signal_id: str = "signal_id"
    timestamp: str = "timestamp"
    value_col: str = "value"
    alpha: float = 0.05
    random_seed: int = None
    ls_range: tuple[float, float] = (10.0, 100.0)
    ls_vals: np.ndarray = None
    n_searches: int = 10
    n_splits: int = 5
    eps: float = 1e-6
    gp_implementation: str = "sklearn"
    sklearn_scoring: str = "neg_mean_squared_error"

    def __post_init__(self):
        valid_methods = {"difference", "detrend"}
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")

        valid_detrend_types = {"lr", "gp"}
        if self.detrend_type not in valid_detrend_types:
            raise ValueError(f"detrend_type must be one of {valid_detrend_types}")

        if not (0 < self.alpha < 1):
            raise ValueError("alpha must be in (0, 1)")

        if self.ls_range[0] <= 0 or self.ls_range[0] >= self.ls_range[1]:
            raise ValueError("ls_range must be a tuple (a, b) with 0 < a < b")

        if not isinstance(self.n_searches, int) or self.n_searches <= 0:
            raise ValueError("n_searches must be a positive integer")

        if not isinstance(self.n_splits, int) or self.n_splits <= 0:
            raise ValueError("n_splits must be a positive integer")

        if self.random_seed is not None and not isinstance(self.random_seed, int):
            raise ValueError("random_seed must be None or an integer.")

        if (
            not (isinstance(self.eps, float) or isinstance(self.eps, int))
            or self.eps < 0
        ):
            raise ValueError("eps must be a non-negative number")

        valid_gp_implementations = {"sklearn", "numba"}
        if self.gp_implementation not in valid_gp_implementations:
            raise ValueError(
                f"gp_implementation must be one of {valid_gp_implementations}"
            )


@dataclass
class SampEnSettings:
    """
    A dataclass to hold the settings for the Sample Entropy computation.

    Attributes:
        signal_id (str): The column name for signal id in the DataFrame. Default is "signal_id".
        timestamp (str): The column name for timestamps in the DataFrame. Default is "timestamp".
        value_col (str): The column name for signal values in the DataFrame. Default is "value".
        n_boot (int): The number of bootstrap samples to use in the estimation. Default is 100.
        n_trials (int): The number of trials for the optimization. Default is 100.
        random_seed (int): The random seed for reproducibility. Default is None (no seeding).
        r_range (tuple): The range of r values for the optimization. Default is (0.10, 0.50).
        m_range (tuple): The range of m values for the optimization. Default is (1, 3).
        p_range (tuple): The range of p values for the stationary bootstrap. Default is (0.01, 0.99).
        lam1 (float): The trade-off parameter between the r-based penalization. Default is 0.33.
        r (float): The user-provided value for r. Default is None (to be optimized).
        m (int): The user-provided value for m. Default is None (to be optimized).
        p (float): The user-provided value for p. Default is None (to be optimized).
    """

    signal_id: str = "signal_id"
    timestamp: str = "timestamp"
    value_col: str = "value"
    n_boot: int = 100
    n_trials: int = 100
    random_seed: int = None
    r_range: tuple = (0.10, 0.50)
    m_range: tuple = (1, 3)
    p_range: tuple = (0.01, 0.99)
    lam1: float = 0.33
    r: float = None
    m: int = None
    p: float = None

    def __post_init__(self):
        self._check_string_attributes("signal_id", "timestamp", "value_col")
        self._check_ranges()
        self._check_fixed_values()
        self._check_positive_integer(self.n_boot, "n_boot")
        self._check_positive_integer(self.n_trials, "n_trials")
        if self.random_seed is not None:
            if not isinstance(self.random_seed, int) or self.random_seed < 0:
                raise ValueError("random_seed must be a non-zero integer")

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
        for arg_name in ["r", "m", "p", "lam1"]:
            arg_value = getattr(self, arg_name)
            if arg_value is not None:
                if arg_name == "m":
                    if not isinstance(arg_value, int) or arg_value <= 0:
                        raise ValueError("m must be an integer > 0.")
                elif arg_name in ["r", "lam1"]:
                    if arg_value <= 0:
                        raise ValueError(f"{arg_name} must be > 0.")
                elif arg_name == "p":
                    if not isinstance(arg_value, float) or not (0 < arg_value < 1):
                        raise ValueError(
                            "p must be a float between 0 and 1 (exclusive)."
                        )

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
