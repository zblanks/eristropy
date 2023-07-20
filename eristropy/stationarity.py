import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.multitest import multipletests

from eristropy.dataclasses import StationarySignalParams
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
        params (StationarySignalParams): The parameters for the transformation.
        stationary_frac_ (float): The fraction of signals that were made stationary
            by the transformation.
        stationary_signals_ (np.ndarray): The IDs of the signals that were made
            stationary by the transformation.
    """

    def __init__(self, df: pd.DataFrame, params: StationarySignalParams = None) -> None:
        self.df = df
        self.stationary_frac_ = None
        self.stationary_signals_ = None

        if params is None:
            self.params = StationarySignalParams()
        elif isinstance(params, StationarySignalParams):
            self.params = params
        else:
            raise TypeError("params must be an instance of StationarySignalParams.")

        _validate_dataframe(
            self.df, self.params.signal_id, self.params.timestamp, self.params.value_col
        )

    def _calculate_pvalues(self) -> np.ndarray:
        """
        Calculates the ADF p-value for each unique signal ID in the DataFrame.

        Returns:
            np.ndarray: Array containing the computed p-values.

        Notes:
            This function uses the augmented Dickey-Fuller test from the
            statsmodels.tsa.stattools module to compute the p-value for each
            unique signal ID in the DataFrame. If a signal is not long enough
            to estimate the p-value, a p-value of 1.0 is assigned.
        """

        # Sort the DataFrame by timestamp
        df = self.df.sort_values(self.params.timestamp)

        # Group the DataFrame by signal ID
        grouped = df.groupby(self.params.signal_id)

        pvalues = []

        for _, group in grouped:
            y = group[self.params.value_col].values.astype(np.float64)
            try:
                pvalue = adfuller(y)[1]
            except ValueError as e:
                # Handle the case where the signal is (probably) not long enough
                pvalue = 1.0
                print(
                    f"An error occurred for group: {group[self.params.signal_id].iloc[0]}"
                )
                print(f"Error message: {str(e)}")
            pvalues.append(pvalue)

        return np.asarray(pvalues)

    def _determine_stationary_signals(self) -> None:
        """
        Compute the fraction of signals in the DataFrame that are statistically stationary.

        This function calculates the fraction of signals in the input DataFrame
        that exhibit stationarity using the Augmented Dickey-Fuller test. It groups
        the DataFrame by signal ID and performs the test for each group. The fraction
        is determined based on the proportion of signals that reject the null hypothesis
        of non-stationarity at the specified significance level.
        """
        pvalues = self._calculate_pvalues()
        multi_rejects = multipletests(pvalues, alpha=self.params.alpha)[0]
        self.stationary_frac_ = multi_rejects.mean()

        unique_signals = self.df[self.params.signal_id].unique()
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
        >>> params = StationarySignalParams(method="difference")
        >>> signals = StationarySignals(df, params)
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
        # Define the dispatch table
        detrend_dispatch = {
            "lr": _detrend_all_signals_linreg,
            "gp": {
                "numba": _detrend_all_signals_gp_numba,
                "sklearn": _detrend_all_signals_gp_sklearn,
            },
        }

        # Handle the case where the method is 'difference'
        if self.params.method == "difference":
            out_df = _difference_all_signals(self.df, self.params)

        # Handle the case where the method is 'detrend'
        elif self.params.method == "detrend":
            detrend_function = detrend_dispatch[self.params.detrend_type]

            if isinstance(detrend_function, dict):  # Handle the gp case
                gp_function = detrend_function[self.params.gp_implementation]

                ls_low, ls_high = self.params.ls_range
                rng = np.random.default_rng(self.params.random_seed)

                self.params.ls_vals = rng.uniform(
                    ls_low, ls_high, size=self.params.n_searches
                )

                out_df = gp_function(self.df, self.params)
            else:  # Handle the lr case
                out_df = detrend_function(self.df, self.params)

        self._determine_stationary_signals()
        stationary_df = out_df.loc[
            out_df[self.params.signal_id].isin(self.stationary_signals_)
        ]

        return stationary_df
