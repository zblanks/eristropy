# StationarySignals

## Overview

The `StationarySignals` class is designed to transform a set of time-series signals into stationary signals. 
This is an essential pre-processing step in many time-series analysis tasks.

## Attributes

- `df`: (`pd.DataFrame`) The input DataFrame containing the signals.
- `signal_id`: (`str`, optional) Column name in `df` containing the signal IDs. Default is 'signal_id'.
- `timestamp`: (`str`, optional) Column name in `df` containing the timestamps. Default is 'timestamp'.
- `value_col`: (`str`, optional) Column name in `df` containing the values. Default is 'value'.
- `method`: (`str`, optional) The method to use for making signals stationary. Default is 'difference'. Choices are 'difference' and 'detrend'.
- `detrend_type`: (`str`, optional) The type of detrending to use if `method` is 'detrend'. Default is 'gp'. Choices are 'gp' and 'lr'.
- `alpha`: (`float`, optional) Significance level for the Augmented Dickey-Fuller test. Default is 0.05.
- `random_seed`: (`int`, optional) Seed for the random number generator. Default is None.
- `ls_range`: (`tuple`, optional) Tuple specifying the range of length-scales for the Gaussian process. Default is (10.0, 100.0).
- `ls_values`: (`np.ndarray`, optional) Array of specific length scale values. Default is None.
- `n_searches`: (`int`, optional) Number of searches for Gaussian process hyperparameters. Default is 10.
- `n_splits`: (`int`, optional) Number of splits for cross-validation. Default is 5.
- `eps`: (`float`, optional) Tolerance for the difference. Default is 1e-6.
- `gp_implementation`: (`str`, optional) Implementation to use for the Gaussian process. Default is 'numba'. Choices are 'sklearn' and 'numba'.
- `sklearn_scoring`: (`str`, optional) Scoring method when using 'sklearn' for Gaussian process. Default is 'neg_mean_squared_error'.
- `normalize_signals`: (`bool`, optional) Whether to normalize signals to zero mean and unit variance. Default is True.

## Methods

### `make_stationary_signals()`

Creates stationary signals at specified statistical level, $\alpha$.

#### Notes
We have implemented an RBF-based GP detrending in Numba using a
standard Cholesky factorization solution. This method, on average,
is faster than the Scikit-Learn implementation. The benefit will
become more noticeable as the number of unique signals increases.
However, this implementation is less numerically stable and less well-tested
than the Scikit-Learn version. You can control which version you use
via the argument `gp_implementation`.

#### Example
```python
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
```
