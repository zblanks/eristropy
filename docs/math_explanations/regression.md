# Regression-Based De-Trending
In EristroPy, we offer two de-trending approaches: linear 
regression and Gaussian processes (GPs).

## Linear Regression
Linear regression is a widely used method for estimating the relationship between 
a time series signal $\mathbf{x} \in \mathbb{R}^N$ of length $N$ and a set of 
predictive features $\mathbf{\Theta} \in \mathbb{R}^{N \times d}$ associated with the signal (e.g., thermodynamic work rate, etc.). The goal is to find the optimal coefficients $\beta^*$ 
that minimize the squared Euclidean norm between the observed signal $\mathbf{x}$ and 
the predicted values $\mathbf{\Theta}\beta$. Mathematically, this is formulated as:

$$
    \beta^* := \text{argmin}_{\beta \in \mathbb{R}^d} \quad \lVert \mathbf{x} - \mathbf{\Theta} \beta \rVert_2^2
$$

Fortunately, there exist incredibly efficient and numerically stable algorithms 
to find the optimal coefficients $\beta^*$. Using the optimal $\beta^*$, we then
de-trend the signal by calculating:

$$
    \widetilde{x}_t := x_t - \theta_t^T \beta^*, \quad t = 1, \ldots, T
$$

However, it's important to note that linear regression is limited to approximating 
trends that can be expressed as a linear combination of the predictive features 
$\mathbf{\Theta}$. In cases where more complex trends are present, such as non-linear 
relationships, linear regression may not provide adequate results. In EristroPy, we 
provide linear regression de-trending in [make_stationary_signals](../api/stationarity.md#make_stationary_signals),
but we do not recommend using this method for de-trending signals.

## Gaussian Process De-Trending in EristroPy
To overcome the limitations of linear regression, EristroPy offers de-trending using 
GPs. Gaussian processes are powerful non-parametric regression 
techniques that model the relationship between inputs and outputs based on the 
assumption of a Gaussian process prior over functions. GPs are particularly 
well-suited for capturing complex, non-linear relationships and providing 
uncertainty estimates. A GP prior is defined as a collection of random variables, 
any finite number of which have a joint Gaussian distribution. It can be thought 
of as a distribution over functions [^1]

In EristroPy, we utilize the radial basis function (RBF) kernel, also known as the 
squared exponential or Gaussian kernel, for GP de-trending. The RBF kernel plays 
a crucial role in capturing the underlying patterns in the data by specifying 
the similarity between input data points.

The RBF kernel between two input points, $\theta$ and $\theta^\prime$ is defined as:

$$
    k(\theta, \theta^\prime) := \exp \left(-\frac{1}{2l^2} \lVert \theta - \theta^\prime \rVert_2^2 \right)
$$

where $l > 0$ defines the length scale hyperparameter, controlling the smoothness of the function. 
The RBF kernel captures the notion that inputs close in the input space should have similar outputs.
The length scale parameter determines how far-reaching the influence of a data point 
is on its neighbors. Additionally, the RBF kernel guarantees that the kernel 
matrix $\mathbf{K} := k(\theta, \theta^\prime) \ \ \forall \theta, \theta^\prime \in \mathbf{\Theta}$ 
is positive definite.

To estimate expected value of a set of new points, $\mathbf{\Theta}_*$, with the associated matrix of 
covariance values $\mathbf{K}_*$, representing the covariance between the new points 
$\mathbf{\Theta}_*$ and all previous samples in $\mathbf{\Theta}$ we calculate:

$$
\mathbb{E}[\mathbf{\Theta}_* \vert l] := \mathbf{K}_*^T \left(\mathbf{K} + \sigma^2 \mathbf{I} \right)^{-1} \mathbf{y}
$$

In the above expression, $\sigma^2 \mathbf{I}$ is added to the kernel matrix to 
account for noise in $\mathbf{y}$ and ensure numerical stability. Instead of 
inverting the matrix, $\mathbf{K} + \sigma^2 \mathbf{I}$, which is an $\mathcal{O}(n^3)$ 
complexity operation, because the jittered kernel matrix is positive definite, 
we can compute the Cholesky factorization of this matrix to achieve superior computational performance.

In EristroPy, the estimation process of the expected value for these new points is roughly
implemented as follows:

```python
K = rbf_kernel(theta, theta, length_scale=l)
Kstar = rbf_kernel(theta_star, theta, length_scale=l)
diag(K) += sigma ** 2
L = scipy.linalg.cho_factor(K, lower=True)
a = scipy.linalg.cho_solve(L, y)
xbar = Kstar.T @ a
```

We determine the optimal length scale value, $l^*$, by performing a randomized
search cross-validation procedure. Finally, using $l^*$, we de-trend the original
time series signal, $\mathbf{y}$, by calculating:

$$
    \widetilde{x}_t := x_t - \mathbb{E}[\theta_t \vert l^*], \quad t = 1, \ldots, T
$$

[^1]:
    Williams, Christopher KI, and Carl Edward Rasmussen. 
    Gaussian processes for machine learning. Vol. 2. No. 3. Cambridge, MA: MIT press, 2006.
