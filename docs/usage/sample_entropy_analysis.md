# Sample Entropy Analysis Example

This section provides a practical guide to conducting a valid entropy analysis using the [`StationarySignals`](../api/stationarity.md) and [`SampleEntropy`](../api/sample_entropy.md) classes in this package. Below are the steps to accomplish this:

## Preliminary Set-Up
```python
import numpy as np
import pandas as pd
from eristropy.stationarity import StationarySiganls
from eristropy.sample_entropy import SampleEntropy
```

Suppose we have time series signals, here we create synthetic signals for demonstration purposes.

```python
signal_ids = np.repeat(["signal_1", "signal_2"], 100)
timestamps = np.tile(np.arange(100), 2)
rng = np.random.default_rng(17)
signal_1_values = rng.uniform(-5, 5, size=(100,))
signal_2_values = rng.uniform(-5, 5, size=(100,))
values = np.concatenate((signal_1_values, signal_2_values))

df = pd.DataFrame({
    "signal_id": signal_ids,
    "timestamp": timestamps,
    "value": values
})
```

## Ensuring Stationary Signals
Before computing SampEn, it's crucial to ensure that the signals are stationary (see [Stationarity](../math_explanations/stationarity.md) for more details). This is where [`StationarySignals`](../api/stationarity.md) comes in handy.

```python
signals = StationarySignals(df, method='difference')
stationary_df = signals.make_stationary_signals()
```

In this example, we use the "differencing"-based approach, but we also allow users to de-trend the signals if desired.

## Find Optimal SampEn Parameters & Compute SampEn
Now, that we have weakly stationary signals, we can compute the SampEn for all unique signals in the dataset. You can also estimate the uncertainty if needed.

```python
sampen = SampleEntropy(stationary_df)
result_df = sampen.compute_all_sampen(optimize=True, estimate_uncertainty=True)
```

The `result_df` will contain SampEn estimates from the optimal $(m, r)$ combination for each unique signal, which you can then analyze further.

```python
print(result_df)
```
