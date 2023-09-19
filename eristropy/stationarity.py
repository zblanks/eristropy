import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.multitest import multipletests

from eristropy.difference import _difference_all_signals
from eristropy.gp import _detrend_all_signals_gp_numba, _detrend_all_signals_gp_sklearn
from eristropy.linreg import _detrend_all_signals_linreg
from eristropy.validation import _validate_dataframe


class StationarySignals:
    """
    Class for making signals stationary.

    This class provides an interface for transforming a set of time-series signals
    into stationary signals. It supports two methods: differencing and detrending.

    Attributes:
        df (pd.DataFrame): The input DataFrame containing the signals.
        method (str, optional): The method to use for making signals stationary.
            Default is 'difference'. Choices are 'difference' and 'detrend'.
        detrend_type (str, optional): The type of detrending to use if method is 'detrend'.
            Default is 'gp'. Choices are 'gp' and 'lr'.
        signal_id (str, optional): Column name in df containing the signal IDs. Default is 'signal_id'.
        timestamp (str, optional): Column name in df containing the timestamps. Default is 'timestamp'.
        value_col (str, optional): Column name in df containing the values. Default is 'value'.
        alpha (float, optional): Significance level for the Augmented Dickey-Fuller test. Default is 0.05.
        random_seed (int, optional): Seed for the random number generator. Default is None.
        ls_range (tuple, optional): Tuple specifying the range of length-scales for the Gaussian process. Default is (10.0, 100.0).
        n_searches (int, optional): Number of searches for Gaussian process hyperparameters. Default is 10.
        n_splits (int, optional): Number of splits for cross-validation. Default is 5.
        eps (float, optional): Tolerance for the difference. Default is 1e-6.
        gp_implementation (str, optional): Implementation to use for the Gaussian process.
            Default is 'numba'. Choices are 'sklearn' and 'numba'.
        sklearn_scoring (str, optional): Scoring method when using 'sklearn' for Gaussian process.
            Default is 'neg_mean_squared_error'.
        normalize_signals (bool, optional): Whether to normalize signals to zero mean and unit variance.
            Default is True.

    Methods:
        _validate_args(): Validates the class attributes.
        _calculate_pvalues(df: pd.DataFrame) -> np.ndarray: Calculates the p-values for stationarity.
        _normalize(x: np.ndarray) -> np.ndarray: Normalizes the signal.
        _determine_stationary_signals(df: pd.DataFrame): Determines which signals are stationary.
        make_stationary_signals() -> pd.DataFrame: Creates stationary signals.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        signal_id: str = "signal_id",
        timestamp: str = "timestamp",
        value_col: str = "value",
        method: str = "difference",
        detrend_type: str = "gp",
        alpha: float = 0.05,
        random_seed: int = None,
        ls_range: tuple[float, float] = (10.0, 100.0),
        n_searches: int = 10,
        n_splits: int = 5,
        eps: float = 1e-6,
        gp_implementation: str = "numba",
        sklearn_scoring: str = "neg_mean_squared_error",
        normalize_signals: bool = True,
    ) -> None:
        self.df = df
        self.signal_id = signal_id
        self.timestamp = timestamp
        self.value_col = value_col
        self.method = method
        self.detrend_type = detrend_type
        self.alpha = alpha
        self.random_seed = random_seed
        self.ls_range = ls_range
        self.n_searches = n_searches
        self.n_splits = n_splits
        self.eps = eps
        self.gp_implementation = gp_implementation
        self.sklearn_scoring = sklearn_scoring
        self.normalize_signals = normalize_signals
        self.stationary_frac_ = None
        self.stationary_signals_ = None

        self._validate_args()
        _validate_dataframe(self.df, self.signal_id, self.timestamp, self.value_col)

        if self.random_seed is None:
            self._rng = np.random.RandomState()
        else:
            self._rng = np.random.RandomState(self.random_seed)

    def _validate_args(self) -> None:
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

    def _calculate_pvalues(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculates the ADF p-value for each unique signal ID in the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame obtained from differencing or detrending

        Returns:
            np.ndarray: Array containing the computed p-values.

        Notes:
            This function uses the augmented Dickey-Fuller test from the
            statsmodels.tsa.stattools module to compute the p-value for each
            unique signal ID in the DataFrame. If a signal is not long enough
            to estimate the p-value, a p-value of 1.0 is assigned.
        """

        # Sort the DataFrame by timestamp
        df = df.sort_values(self.timestamp)

        # Group the DataFrame by signal ID
        grouped = df.groupby(self.signal_id)

        pvalues = []

        for _, group in grouped:
            y = group[self.value_col].values.astype(np.float64)
            try:
                pvalue = adfuller(y)[1]
            except ValueError as e:
                # Handle the case where the signal is (probably) not long enough
                pvalue = 1.0
                print(f"An error occurred for group: {group[self.signal_id].iloc[0]}")
                print(f"Error message: {str(e)}")
            pvalues.append(pvalue)

        return np.asarray(pvalues)

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        """Normalizes the input signal to zero mean and unit variance"""
        return (x - x.mean()) / x.std()

    def _determine_stationary_signals(self, df: pd.DataFrame) -> None:
        """
        Compute the fraction of signals in the DataFrame that are statistically stationary.

        This function calculates the fraction of signals in the input DataFrame
        that exhibit stationarity using the Augmented Dickey-Fuller test. It groups
        the DataFrame by signal ID and performs the test for each group. The fraction
        is determined based on the proportion of signals that reject the null hypothesis
        of non-stationarity at the specified significance level.

        Args:
            df: DataFrame obtained after differencing or detrending
        """
        pvalues = self._calculate_pvalues(df)
        multi_rejects = multipletests(pvalues, alpha=self.alpha)[0]
        self.stationary_frac_ = multi_rejects.mean()

        unique_signals = self.df[self.signal_id].unique()
        self.stationary_signals_ = unique_signals[multi_rejects]

    def make_stationary_signals(self) -> pd.DataFrame:
        """
        Create stationary signals using the specified method.

        Returns:
            pd.DataFrame: The DataFrame with stationary signals.

        Notes:
            We have implemented an RBF-based GP detrending in Numba using a
            standard Cholesky factorization solution. This method, on average,
            is faster than the Scikit-Learn implementation. The benefit will
            become more noticeable as the number of unique signals, $N$, increases.
            However, this implementation is less numerically stable and less well-tested
            than the Scikit-Learn version. You can control which version you use
            via the argument `gp_implementation` in the StationarySignalsParams
            dataclass.

        Example:
        >>> signal_ids = np.repeat(["abc", "def"], 100)
        >>> timestamps = np.tile(np.arange(100), 2)
        >>> rng = np.random.default_rng(17)
        >>> abc_values = rng.uniform(-5, 5, size=(100,))
        >>> def_values = rng.uniform(-5, 5, size=(100,))
        >>> values = np.concatenate((abc_values, def_values))
        >>> df = pd.DataFrame({
        ...     "signal_id": signal_ids,
        ...     "timestamp": timestamps,
        ...     "value": values
        ... })
        >>> signals = StationarySignals(df, method='difference', normalize_signals=False)
        >>> signals.make_stationary_signals()
            signal_id  timestamp     value
        0         abc          1 -6.841017
        1         abc          2  3.967715
        2         abc          3 -1.896646
        3         abc          4 -1.531380
        4         abc          5  1.708821
        ..        ...        ...       ...
        193       def         95  0.653840
        194       def         96  0.846767
        195       def         97  5.441443
        196       def         98 -8.955780
        197       def         99  5.397502
        """

        # Handle the case where the method is 'difference'
        if self.method == "difference":
            out_df = _difference_all_signals(
                self.df, self.signal_id, self.timestamp, self.value_col
            )

        # Handle the case where the method is 'detrend'
        elif self.method == "detrend":
            if self.detrend_type == "lr":
                out_df = _detrend_all_signals_linreg(
                    self.df, self.signal_id, self.timestamp, self.value_col
                )
            elif self.detrend_type == "gp":
                if self.gp_implementation == "numba":
                    out_df = _detrend_all_signals_gp_numba(
                        self.df,
                        self.signal_id,
                        self.timestamp,
                        self.value_col,
                        self._rng,
                        self.ls_range,
                        self.n_searches,
                        self.n_splits,
                        self.eps,
                    )
                elif self.gp_implementation == "sklearn":
                    out_df = _detrend_all_signals_gp_sklearn(
                        self.df,
                        self.signal_id,
                        self.timestamp,
                        self.value_col,
                        self._rng,
                        self.ls_range,
                        self.sklearn_scoring,
                    )

        self._determine_stationary_signals(out_df)
        stationary_df = out_df.loc[
            out_df[self.signal_id].isin(self.stationary_signals_)
        ].copy(deep=True)

        if self.normalize_signals:
            stationary_df[self.value_col] = stationary_df.groupby(self.signal_id)[
                self.value_col
            ].transform(self._normalize)

        return stationary_df
