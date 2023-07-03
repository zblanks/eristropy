# Stationarity

Welcome to the documentation for the Stationarity module within the CPyET package. 
This module provides functions for constructing stationary CPET signals, 
and determining the proportion of stationary signals in a given dataset.

## Functions

### [`make_stationary_signals`](#make-stationary-signals)
Create stationary signals using the specified method.

#### Description
This function takes an input DataFrame containing signals and applies a specified method 
to make the signals stationary. Stationarity is a necessary condition for valid entropy
and variability analysis. The function either de-trends or differences a given
signal to, ideally, yield statistically stationary signals.

#### Syntax
```python
make_stationary_signals(
    df: pd.DataFrame,
    method: str,
    detrend_type: str = "gp",
    signal_id: str = "signal_id",
    timestamp: str = "timestamp",
    value_col: str = "value",
    alpha: float = 0.05,
    random_seed: int = None,
    ls_range: tuple[float, float] = (1.0, 100.0),
    n_searches: int = 10,
    n_splits: int = 5,
    eps: float = 1e-6,
    gp_implementation: str = "sklearn",
) -> pd.DataFrame:
```

#### Args
- `df` (pd.DataFrame): The input DataFrame containing the signals.
- `method` (str): The method to make the signals stationary. Valid options are 'difference' and 'detrend'.
- `detrend_type` (str, optional): The type of detrending. Required when the method is 'detrend'. Valid options are 'lr' and 'gp'. Default is 'gp'.
- `signal_id` (str, optional): The column name for the signal ID. Default is 'signal_id'.
- `timestamp` (str, optional): The column name for the timestamp. Default is 'timestamp'.
- `value_col` (str, optional): The column name for the signal values. Default is 'value'.
- `alpha` (float, optional): The significance level for the stationarity test. Must be in the range (0, 1). Default is 0.05.
- `random_seed` (int, optional): The random seed for generating ls_vals. Must be an integer. Default is None.
- `ls_range` (tuple[float, float], optional): The range of ls values for GP detrending. The lower bound must be greater than 0. Default is (1.0, 100.0).
- `n_searches` (int, optional): The number of ls values to consider for GP detrending. Must be a positive integer. Default is 10.
- `n_splits` (int, optional): The number of cross-validation splits. Must be a positive integer. Default is 5.
- `eps` (float, optional): Small value added to the kernel matrix for numerical stability. Default is 1e-6.
- `gp_implementation` (str, optional): Which GP detrending to use when the method is 'detrend' and detrend_type is 'gp'. Valid options are 'sklearn' and 'numba'. Default is 'sklearn'.

#### Returns
* `pd.DataFrame`: The DataFrame with stationary signals.

#### Notes
We have implemented an RBF-based GP detrending in Numba using a
standard Cholesky factorization solution. This method, on average,
is faster than the Scikit-Learn implementation. The benefit will
become more noticeable as the number of unique signals, \(N\), increases.
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
>>> make_stationary_signals(df, method="difference")
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

### `determine_stationary_signals`
Compute the fraction and unique identifiers of signals in the DataFrame that 
are statistically stationary.

#### Syntax
```python
def determine_stationary_signals(
    df: pd.DataFrame,
    alpha: float = 0.05,
    signal_id: str = "signal_id",
    timestamp: str = "timestamp",
    value_col: str = "value",
) -> tuple[float, np.ndarray]:
```

#### Args
* `df` (pd.DataFrame): The input DataFrame containing signal observations.
* `alpha` (float, optional): The significance level for the stationarity test (default: 0.05).
* `signal_id` (str, optional): The column name for the signal ID (default: 'signal_id').
* `timestamp` (str, optional): The column name for the timestamp (default: 'timestamp').
* `value_col` (str, optional): The column name for the signal values (default: 'value').

#### Returns
* `tuple[float, np.ndarray]`: The fraction of signals that are statistically 
stationary and array of signal IDs which are statistically stationary

#### Example
```python
>>> import numpy as np
>>> import pandas as pd
>>> from cypet.stationarity import determine_stationary_signals
>>> signal_ids = np.repeat(["abc", "def"], 100)
>>> timestamps = np.tile(np.arange(100), 2)
>>> abc_values = np.linspace(0, 100, 100)
>>> def_values = np.sin(np.linspace(0, 2 * np.pi, 100))
>>> values = np.concatenate((abc_values, def_values))
>>> df = pd.DataFrame({
...     "signal_id": signal_ids,
...     "timestamp": timestamps,
...     "value": values
... })
>>> stationary_fraction, stationary_signals = determine_stationary_signals(df)
>>> print(stationary_fraction)
0.5
>>> print(stationary_signals)
array(["def"])
```
