# Stationarity

## What is stationarity?
In the context of time series analysis, a stationary time series refers to a 
series of observations or data points where statistical properties remain constant over time. 
In other words, the behavior and characteristics of the time series do not change as time progresses.

There are two main components to consider when determining if a time series is stationary:

* **Constant Mean**: The mean (average) of the time series remains constant over time. 
This implies that, on average, the observations do not exhibit any long-term trend 
or systematic upward or downward shifts.

* **Constant Variance**: The variance (or standard deviation) of the time series 
remains constant over time. This indicates that the dispersion or spread of the 
observations around the mean does not change with time. It implies that the 
fluctuations or variability of the series do not exhibit any systematic change.

Technically speaking, the two above properties, if true, define a *weakly stationary* signal.
To have a *strictly stationary* signal, it is also necessary to have
constant autocovariance. However, for the purposes of CPET variability analysis,
this condition is not necessary, and so when we say that a signal is stationary,
we mean in the weak sense.

## Why do we care if a signal is stationary?
In this package, we primarily refer to variability anlaysis in the context of entropy,
specifically sample entropy and permutation entropy. Richman and Moorman[^1],
the researchers who designed sample entropy, and Bandt and Pompe[^2] (permutation entropy),
both assumed as a starting point that the input signal was stationary. Moreover,
Chatain et al.[^3] (and others) have done empirical work demonstrating why
stationarity is a necessary condition for valid time series variability analysis.
Practicaly speaking, if the signal is non-stationary, such as having an 
increasing mean during CPET, it is likely to result in higher entropy values due 
to the template matching scheme based on a fixed radial distance 
(see [Entropy](entropy.md) for further explanation on what this means).
When it comes to addressing non-stationarity in signals, specifically when the 
main issue is a non-constant mean (the most common issue with CPET signals), 
there are two common approaches: differencing and de-trending.[^4]

## Differencing
Differencing involves creating a new signal by subtracting the previous observation 
from the current observation. Let $\mathbf{y} \in \mathbb{R}^T$ define a time series
signal of length, $T$. Formally, differencing is defined as:

$$
    \tilde s_t := y_t - y_{t-1}, \quad t = 2, \ldots, T
$$

The rationale behind differencing stems from autoregressive processes and random walk theory.[^5]
This approach is relatively straightforward to implement 
(see: [make_stationary_signals](../api/stationarity.md#make-stationary-signals) for details),
and one can assess the statistical stationarity of the differenced signal, 
$\tilde{\mathbf{s}}$, at a given significance level, $\alpha$, using the 
Augmented Dickey-Fuller (ADF) test.[^6]

## De-Trending
Alternatively, one can estimate and remove the trend present in a non-stationary signal.
This approach offers various implementations, with linear or polynomial regression 
being the most common.[^7] Any algorithm that estimates $\hat{y}_t := f(\mathbf{x}_t)$,
where $\mathbf{x}_t$ represents a set of predictive features and $f(\cdot)$ is a regression function,
can be used to de-trend the original signal, $\mathbf{y}$, by calculating:

$$
    \hat{s}_t := y_t - \hat{y}_t, \quad t = 1, \ldots, T
$$

In CPyET, we provide two options for de-trending: a standard linear regression function
and a radial basis function Gaussian process (GP). If you need to de-trend a 
non-stationary signal, we highly recommend using the GP implementation. The GP 
method provided in [make_stationary_signals](../api/stationarity.md#make_stationary_signals) 
has the advantage of automatically accommodating nonlinear trends. Our empirical 
results demonstrate that it significantly increases the proportion of statistically 
stationary signals. For a more in depth discussion of the mathematical aspects of 
linear regression and GPs, please refer to the [Regression](regression.md) section.

## Should I choose differencing or de-trending?
Both methods are implemented in [make_stationary_signals](../api/stationarity.md#make_stationary_signals)
and have strong theoretical and empirical bases. In practice, we recommend trying
both approaches and seeing which method yields a larger proportion of statistically stationary
signals.


[^1]:
    Richman, Joshua S., and J. Randall Moorman. "Physiological time-series analysis using approximate entropy and sample entropy." 
    American journal of physiology-heart and circulatory physiology 278.6 (2000): H2039-H2049.
[^2]:
    Bandt, Christoph, and Bernd Pompe. "Permutation entropy: a natural complexity measure for time series." 
    Physical review letters 88.17 (2002): 174102.
[^3]:
    Chatain, Cyril, et al. "Effects of nonstationarity on muscle force signals regularity during a fatiguing motor task." 
    IEEE Transactions on Neural Systems and Rehabilitation Engineering 28.1 (2019): 228-237.
[^4]: 
    Berry, Nathaniel T., et al. "Heart rate dynamics during acute recovery from maximal aerobic exercise in young adults." 
    Frontiers in Physiology 12 (2021): 627320.
[^5]:
    Gonedes, Nicholas J., and Harry V. Roberts. "Differencing of random walks and near random walks." 
    Journal of Econometrics 6.3 (1977): 289-308.
[^6]:
    Dickey, David A., and Wayne A. Fuller. "Distribution of the estimators for autoregressive time series with a unit root." 
    Journal of the American statistical association 74.366a (1979): 427-431.
[^7]:
    Berry, Nathaniel T., Laurie Wideman, and Christopher K. Rhea. "Variability and complexity of non-stationary functions: methods for post-exercise HRV." 
    Nonlinear Dynamics, Psychology & Life Sciences 24.4 (2020).