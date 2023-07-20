import dataclasses

import numpy as np


@dataclasses.dataclass
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
