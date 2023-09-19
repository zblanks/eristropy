# EristroPy: End-to-End Entropy Analysis of Time Series Signals

## Overview and Introduction

Welcome to EristroPy, a powerful Python package designed for end-to-end entropy analysis of time series signals via entropy. EristroPy provides an all-in-one solution for researchers and practitioners looking to perform entropy/variability analysis of time series data.

For more detailed information, check out the [documentation](https://zblanks.github.io/eristropy/).

## Features & Benefits

EristroPy offers a multitude of features aimed at simplifying the time series analysis process:

- **Automatic Signal Stationarity**: Ensure the validity of your entropy and variability analyses by automatically rendering signals stationary. 
- **Scalable Entropy Calculations**: Utilizes Numba's just-in-time compilation for efficient sample and permutation entropy calculations. 
- **Optimal Parameter Selection**: Provides intelligent suggestions for entropy measure parameters through rigorous statistical approaches.

## Installation

The easiest way to install EristroPy is using pip by calling:

```bash
pip install eristropy
```

## Usage
Using EristroPy, you can go from having the base time series signals, to a coherent & optimized entropy estimates in just a few lines of code. For instance, consider the following problem of estimating the sample entropy of some synthetic time series signals:

```python
import numpy as np
import pandas as pd
from eristropy.stationarity import StationarySignals
from eristropy.sample_entropy import SampleEntropy

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

signals = StationarySignals(df, method="difference")
stationary_df = signals.make_stationary_signals()

sampen = SampleEntropy(stationary_df)
result_df = sampen.compute_all_sampen(optimize=True, estimate_uncertainty=True)
```

In just a few lines of code, you have access to state-of-the-art results that follows & implements best practices. It's that easy!


## License
EristroPy is released under the MIT License. See the [LICENSE](LICENSE.md) file for more details.
