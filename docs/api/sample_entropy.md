# SampleEntropy

## Overview
The `SampleEntropy` class computes the Sample Entropy (SampEn) of multiple signals and finds the optimal SampEn parameters using a regularized MSE objective function and the Optuna Bayesian optimization framework with the Tree-based Parzen Estimator (TPE) surrogate function. 

For more details on this optimization procedure, see Z. Blanks *et al.*, Optimal Sample Entropy Parameter Selection for Short Time Series Signals via Bayesian Optimization, 2023.

The main methods are [`find_optimal_sampen_params()`](../api/sample_entropy.md#find_optimal_sampen_params) for performing the optimization and [`compute_all_sampen()`](../api/sample_entropy.md#compute_all_sampen) for computing the SampEn of all signals with the optimized or user-provided parameters.

## Attributes
- `df`: (`pd.DataFrame`) The DataFrame containing the signals. Must contain columns for the signal_id, timestamp, and signal value at each timestamp.
- `signal_id`: (`str`, optional) Column name in `df` containing the signal IDs. Default is 'signal_id'.
- `timestamp`: (`str`, optional) Column name in `df` containing the timestamps. Default is 'timestamp'.
- `value_col`: (`str`, optional) Column name in `df` containing the values. Default is 'value'.
- `objective`: (`str`, optional) Objective function to minimize. Default is 'mse'. Choices are 'mse' and 'sampen_eff'.
- `n_boot`: (`int`, optional) Number of bootstrap samples to use in the estimation. Default is 100.
- `n_trials`: (`int`, optional) Number of trials for the optimization. Default is 100.
- `random_seed`: (`int`, optional) Seed for the random number generator. Default is None.
- `r_range`: (`tuple[float, float]`, optional) Tuple specifying the range of $r$ values for the optimization. Default is (0.10, 0.50).
- `m_range`: (`tuple[int, int]`, optional) Tuple specifying the range of $m$ values for the optimization. Default is (1, 3).
- `p_range`: (`tuple[float, float]`, optional) Tuple specifying the range of $p$ values for the stationary bootstrap. Default is (0.01, 0.99).
- `lam`: (`float`, optional) The trade-off parameter between the $r$-based penalization. Default is 0.33.
- `r`: (`float`, optional) User-provided value for $r$. Default is None.
- `m`: (`int`, optional) User-provided value for $m$. Default is None.
- `p`: (`float`, optional) User-provided value for $p$. Default is None.

## Methods

### `find_optimal_sampen_params`
Finds the optimal $(m, r)$ SampEn parameters for the input signal set.

#### Notes
This method uses the Optuna library for optimizing the parameters $(m, r, p)$ using a TPE surrogate function.

#### Example
```python
>>> signals = SampleEntropy(df, n_trials=50)
>>> signals.find_optimal_sampen_params()
```

### `compute_all_sampen`
Computes the SampEn of the input signal set given either the provided or optimized values of $(m, r)$.

#### Parameters
- `optimize`: (`bool`, optional) If True, optimize the SampEn parameters before computing the SampEn for all signals. Defaults to False.
- `estimate_uncertainty`: (`bool`, optional) If True, estimates the SE(SampEn) for the given or optimized $(m, r)$ values. Defaults to False.

#### Returns
- `pd.DataFrame`: SampEn estimates given $(m, r)$ for all signals in the data.

#### Example
```python
>>> signals = SampleEntropy(df)
>>> results = signals.compute_all_sampen(optimize=True)
```

### `get_optimization_results`
Return a DataFrame of the optimization results.

#### Parameters
- `attrs`: (tuple, optional) Attributes of the optuna trials to include in the dataframe. By default it includes the trial number, value of the objective function, and parameters used. Refer to optuna documentation for other options.

#### Returns
- `pd.DataFrame`: DataFrame of the optimization trials.

#### Example
```python
>>> signals = SampleEntropy(df)
>>> signals.find_optimal_sampen_params()
>>> optimization_results = signals.get_optimization_results()
```
