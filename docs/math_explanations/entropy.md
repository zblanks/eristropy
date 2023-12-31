# Entropy

## Shannon Entropy
Information theory is a discipline intersecting mathematics and computer science that focuses on the quantification, storage, and communication of information. One of its fundamental tenets, entropy, was introduced by Claude Shannon in 1948[^1]. It quantifies the average amount of information or uncertainty inherent in a random variable or data source. If we consider a discrete random variable, $X$, with support $S_X$ and a probability mass function, $p(x)$, the entropy of this random variable, $X$, can be calculated as:

$$
    H(X) = -\sum_{x \in S_X} p(x) \log_2 p(x)
$$

Higher entropy equates to greater uncertainty, and lower entropy implies increased predictability.

## Entropy of Time Series Signals
In Shannon's groundbreaking work, he introduced a rigorous definition of uncertainty for *static* systems. However, time series signals represent dynamic systems that evolve over time due to perturbations and inputs. Therefore, it is crucial to develop a concept of entropy that captures the complexity and uncertainty of these dynamic systems.

In 1958 and 1959, Kolmogorov and Sinai developed such a measure, now coined Kolmogorov-Sinai (KS) entropy[^2]. KS entropy is a measure of the complexity or randomness present in a dynamical system or time series. It quantifies the rate at which information is generated by the system over time. While KS entropy provides a rigorous and mathematically precise definition of entropy applied to time series signals, it is often not a computationally tractable one[^3]. It is for this reason that researchers have developed alternate measures, such as sample entropy (SampEn) and permutation entropy (PermEn).

## Sample Entropy
SampEn is a measure used to estimate the complexity of time series data. Developed by Richman and Moorman[^4], it builds upon Pincus's approximate entropy[^5]. The concept of SampEn revolves around the idea of template matching in a signal. A template is a subset of consecutive data points from the signal.

Define a time series signal of length $N$ as $\mathbf{x} \in \mathbb{R}^N$. A template of length $m$ (known as the "embedding dimension") is denoted by $\mathbf{u}_m(i) = (x_i, \ldots, x_{i + m - 1})$. We compare these templates and consider them a match if the distance between them is less than a defined radius, $r$. Specifically, a match occurs when $\lVert \mathbf{u}_m(i) - \mathbf{u}_m(j)\rVert_p \leq r$, where $p$ is the order of the norm (typically $p = \infty$ for the L-infinity norm denoting the maximum absolute difference between the elements of two templates) and $r > 0$ is the predefined radius. Unlike ApEn, SampEn does not consider self-matches.

To calculate signal regularity, we compute two quantities, $B^m(r)$ and $A^m(r)$. $B^m(r)$, defined as:

$$
    B^m(r) = \frac{1}{Z(N, m)} \sum_{i = 1}^{N-m} \sum_{\substack{j=1\ j\neq i}}^{N-m} \mathbf{1}\left[\lVert \mathbf{u}_m(i) - \mathbf{u}_m(j)\rVert_p \leq r\right],
$$

is the probability of the signal remaining within a radius, $r$, for $m$ steps. Similarly, $A^m(r)$, given by:

$$
    A^m(r) = \frac{1}{Z(N, m)} \sum_{i = 1}^{N-m} \sum_{\substack{j=1\ j\neq i}}^{N-m} \mathbf{1}\left[\lVert\mathbf{u}_{m+1}(i) - \mathbf{u}_{m+1}(j)\rVert_p \leq r\right],
$$

is the probability that the signal stays within the same radius for an additional step ($m + 1$ steps in total). In both expressions, $\mathbf{1}\left[\cdot \right]$ denotes the indicator function and $Z(N, m)$ is a normalization constant to ensure valid probabilities. With these values, we define SampEn as the negative logarithm of the conditional probability that a sequence will remain within radius $r$ for $m + 1$ steps, given it has stayed within $r$ for $m$ steps:

$$
    \text{SampEn}(\mathbf{x}, m, r) = -\log\left( \frac{A^m(r)}{B^m(r)}\right).
$$

However, if no matches are found at radius $r$ across the signal (i.e., $B^m(r) = 0$), $\text{SampEn}(\mathbf{x}, m, r)$ is undefined.

[^1]:
Shannon, Claude Elwood. "A mathematical theory of communication." The Bell system technical journal 27.3 (1948): 379-423.

[^2]:
Shiryayev, A. N. "New metric invariant of transitive dynamical systems and automorphisms of Lebesgue spaces." Selected Works of AN Kolmogorov: Volume III: Information Theory and the Theory of Algorithms (1993): 57-61.

[^3]:
Kantz, Holger, and Thomas Schreiber. Nonlinear time series analysis. Vol. 7. Cambridge university press, 2004.

[^4]:
Richman, Joshua S., and J. Randall Moorman. "Physiological time-series analysis using approximate entropy and sample entropy." American journal of physiology-heart and circulatory physiology 278.6 (2000): H2039-H2049.

[^5]:
Pincus, Steven M. "Approximate entropy as a measure of system complexity." Proceedings of the National Academy of Sciences 88.6 (1991): 2297-2301.
