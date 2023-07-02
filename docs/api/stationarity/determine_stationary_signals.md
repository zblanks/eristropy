# determine_stationary_signals

Compute the fraction of signals in the DataFrame that are statistically stationary.

## Syntax
```python
def determine_stationary_signals(
    df: pd.DataFrame,
    alpha: float = 0.05,
    signal_id: str = "signal_id",
    timestamp: str = "timestamp",
    value_col: str = "value",
) -> tuple[float, np.ndarray]:
```

## Args
* `df` (pd.DataFrame): The input DataFrame containing signal observations.
* `alpha` (float, optional): The significance level for the stationarity test (default: 0.05).
* `signal_id` (str, optional): The column name for the signal ID (default: 'signal_id').
* `timestamp` (str, optional): The column name for the timestamp (default: 'timestamp').
* `value_col` (str, optional): The column name for the signal values (default: 'value').

## Returns
* `tuple[float, np.ndarray]`: The fraction of signals that are statistically 
stationary and array of signal IDs which are statistically stationary

## Raises
ValueError: If the required columns are missing in the DataFrame,
    if the DataFrame is empty,
    if the significance level is invalid,
    if the signal is not long enough for the ADF test,
    or if no groups are found after grouping by signal ID.

## Example
```python
import numpy as np
import pandas as pd
from cpyet.stationarity import determine_stationary_signals

signal_ids = np.repeat(["abc", "def"], 100)
timestamps = np.tile(np.arange(100), 2)
abc_values = np.linspace(0, 100, 100)
def_values = np.sin(np.linspace(0, 2 * np.pi, 100))
values = np.concatenate((abc_values, def_values))
df = pd.DataFrame({
    "signal_id": signal_ids,
    "timestamp": timestamps,
    "value": values
})

stationary_fraction, stationary_signals = determine_stationary_signals(df)
print(stationary_fraction)
print(stationary_signals)
```

Output:

```python
[0.5]
['def']
```