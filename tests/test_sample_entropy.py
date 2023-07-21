import numpy as np
import pandas as pd
import pytest

from eristropy.dataclasses import SampEnSettings
from eristropy.sample_entropy import OptimizationFailureWarning, SampleEntropy


def test_bootstrap_mse():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(200))

    # Test with different m, r, p, and n_boot values
    m_values = [1, 2, 3]
    r_values = [0.1, 0.2, 0.3]
    p_values = [0.1, 0.5, 0.9]
    n_boot_values = [10, 50, 100]

    for m in m_values:
        for r in r_values:
            for p in p_values:
                for n_boot in n_boot_values:
                    mse = SampleEntropy._bootstrap_mse(x, m, r, p, n_boot)
                    assert isinstance(mse, (float, np.float64))
                    if not np.isnan(mse):
                        assert mse >= 0


def test_bootstrap_mse_constant_signal():
    x = np.full(200, 0.5)
    m, r, p, n_boot = 2, 0.2, 0.5, 50
    mse = SampleEntropy._bootstrap_mse(x, m, r, p, n_boot)
    np.testing.assert_almost_equal(mse, 0, decimal=2)


def test_bootstrap_mse_linear_signal():
    # The MSE should be > 0 and >= the constant signal MSE
    x = np.linspace(-1, 1, 200)
    m, r, p, n_boot = 2, 0.2, 0.5, 50
    mse_linear = SampleEntropy._bootstrap_mse(x, m, r, p, n_boot)
    assert mse_linear >= 0.0

    y = np.full(200, 0.5)
    mse_constant = SampleEntropy._bootstrap_mse(y, m, r, p, n_boot)
    assert mse_linear >= mse_constant


@pytest.mark.parametrize("n1,n2", [(50, 100), (100, 200), (200, 400)])
def test_bootstrap_mse_varying_signal_lengths(n1, n2):
    # MSE should decrease with longer signals
    rng = np.random.default_rng(17)
    m, r, p, n_boot = 1, 0.2, 0.5, 50
    x1 = rng.normal(size=n1)
    x2 = rng.normal(size=n2)
    mse1 = SampleEntropy._bootstrap_mse(x1, m, r, p, n_boot)
    mse2 = SampleEntropy._bootstrap_mse(x2, m, r, p, n_boot)
    assert mse2 <= mse1


@pytest.mark.parametrize("n_boot1,n_boot2", [(50, 250), (100, 500)])
def test_bootstrap_mse_increasing_bootstrap_samples(n_boot1, n_boot2):
    # MSE should decrease with more bootstrap samples
    rng = np.random.default_rng(17)
    x = rng.normal(size=200)
    m, r, p = 1, 0.2, 0.5
    mse1 = SampleEntropy._bootstrap_mse(x, m, r, p, n_boot1)
    mse2 = SampleEntropy._bootstrap_mse(x, m, r, p, n_boot2)
    assert mse2 <= mse1


def test_sample_entropy_init_valid():
    rng = np.random.default_rng(17)
    df = pd.DataFrame(
        {
            "signal_id": np.repeat(range(5), 200),
            "timestamp": np.tile(range(200), 5),
            "value": rng.normal(size=1000),
        }
    )

    sampen = SampleEntropy(df, SampEnSettings(n_trials=10, random_seed=17))
    assert isinstance(sampen, SampleEntropy)
    assert sampen.df.equals(df)
    assert isinstance(sampen.params, SampEnSettings)


def test_sample_entropy_init_default():
    rng = np.random.default_rng(17)
    df = pd.DataFrame(
        {
            "signal_id": np.repeat(range(5), 200),
            "timestamp": np.tile(range(200), 5),
            "value": rng.normal(size=1000),
        }
    )

    sampen = SampleEntropy(df)
    assert isinstance(sampen, SampleEntropy)
    assert sampen.df.equals(df)
    assert isinstance(sampen.params, SampEnSettings)


def test_sample_entropy_init_bad_param_type():
    rng = np.random.default_rng(17)
    df = pd.DataFrame(
        {
            "signal_id": np.repeat(range(5), 200),
            "timestamp": np.tile(range(200), 5),
            "value": rng.normal(size=1000),
        }
    )

    with pytest.raises(TypeError):
        SampleEntropy(df, "not a SampEnSettings object")


def test_sample_entropy_init_optimization_vars():
    rng = np.random.default_rng(17)
    df = pd.DataFrame(
        {
            "signal_id": np.repeat(range(5), 200),
            "timestamp": np.tile(range(200), 5),
            "value": rng.normal(size=1000),
        }
    )
    settings = SampEnSettings(m=2, r=0.2, p=0.5)
    sampen = SampleEntropy(df, settings)
    assert sampen.m_star_ == settings.m
    assert sampen.r_star_ == settings.r
    assert sampen.p_star_ == settings.p


def test_sample_entropy_init_signal_groups():
    rng = np.random.default_rng(17)
    df = pd.DataFrame(
        {
            "signal_id": np.repeat(range(5), 200),
            "timestamp": np.tile(range(200), 5),
            "value": rng.normal(size=1000),
        }
    )
    sampen = SampleEntropy(df, SampEnSettings())
    assert np.array_equal(sampen.unique_signals_, df["signal_id"].unique())
    assert isinstance(sampen.signal_groups_, pd.core.groupby.generic.DataFrameGroupBy)


def test_sample_entropy_init_m_range_too_large():
    rng = np.random.default_rng(17)
    values = rng.normal(size=1000)
    df = pd.DataFrame(
        {
            "signal_id": np.repeat(range(5), 200),
            "timestamp": np.tile(range(200), 5),
            "value": values,
        }
    )

    # m_range high end is too large for the shortest time series
    settings = SampEnSettings(m_range=(1, 250))

    with pytest.raises(
        ValueError,
        match=(
            "The upper limit of m_range cannot exceed the length "
            "of the shortest time series in the DataFrame."
        ),
    ):
        SampleEntropy(df, settings)


def test_find_optimal_sampen_params():
    rng = np.random.default_rng(17)
    df = pd.DataFrame(
        {
            "signal_id": np.repeat(range(5), 200),
            "timestamp": np.tile(range(200), 5),
            "value": rng.normal(size=1000),
        }
    )

    # Initialize SampleEntropy
    params = SampEnSettings(n_trials=10, random_seed=17)
    sampen = SampleEntropy(df, params)

    # Ensure m_star_, r_star_, p_star_ are None before optimization
    assert sampen.m_star_ is None
    assert sampen.r_star_ is None
    assert sampen.p_star_ is None

    # Run find_optimal_sampen_params
    sampen.find_optimal_sampen_params()

    # Check that m_star_, r_star_, p_star_ have been updated
    assert sampen.m_star_ is not None
    assert sampen.r_star_ is not None
    assert sampen.p_star_ is not None

    # Check that m_star_, r_star_, p_star_ fall within the specified ranges
    assert sampen.params.m_range[0] <= sampen.m_star_ <= sampen.params.m_range[1]
    assert sampen.params.r_range[0] <= sampen.r_star_ <= sampen.params.r_range[1]
    assert sampen.params.p_range[0] <= sampen.p_star_ <= sampen.params.p_range[1]


def test_find_optimal_sampen_params_varying_lengths():
    rng = np.random.default_rng(17)
    signal_1 = np.arange(200)
    signal_2 = np.arange(150)
    signal_3 = np.arange(250)

    df = pd.DataFrame(
        {
            "signal_id": np.concatenate(
                [
                    np.repeat(0, len(signal_1)),
                    np.repeat(1, len(signal_2)),
                    np.repeat(2, len(signal_3)),
                ]
            ),
            "timestamp": np.concatenate([signal_1, signal_2, signal_3]),
            "value": rng.normal(size=len(signal_1) + len(signal_2) + len(signal_3)),
        }
    )

    # Initialize SampleEntropy
    sampen = SampleEntropy(df, SampEnSettings(n_trials=10, random_seed=17))

    # Ensure m_star_, r_star_, p_star_ are None before optimization
    assert sampen.m_star_ is None
    assert sampen.r_star_ is None
    assert sampen.p_star_ is None

    # Run find_optimal_sampen_params
    sampen.find_optimal_sampen_params()

    # Check that m_star_, r_star_, p_star_ have been updated
    assert sampen.m_star_ is not None
    assert sampen.r_star_ is not None
    assert sampen.p_star_ is not None

    # Check that m_star_, r_star_, p_star_ fall within the specified ranges
    assert sampen.params.m_range[0] <= sampen.m_star_ <= sampen.params.m_range[1]
    assert sampen.params.r_range[0] <= sampen.r_star_ <= sampen.params.r_range[1]
    assert sampen.params.p_range[0] <= sampen.p_star_ <= sampen.params.p_range[1]


def test_find_optimal_sampen_params_short_signals():
    rng = np.random.default_rng(17)
    signal_1 = np.arange(15)
    signal_2 = np.arange(15)

    df = pd.DataFrame(
        {
            "signal_id": np.concatenate(
                [np.repeat(0, len(signal_1)), np.repeat(1, len(signal_2))]
            ),
            "timestamp": np.concatenate([signal_1, signal_2]),
            "value": rng.normal(size=len(signal_1) + len(signal_2)),
        }
    )

    # Initialize SampleEntropy
    sampen = SampleEntropy(df, SampEnSettings(n_trials=10, random_seed=17))

    # Ensure m_star_, r_star_, p_star_ are None before optimization
    assert sampen.m_star_ is None
    assert sampen.r_star_ is None
    assert sampen.p_star_ is None

    # Check that OptimizationFailureWarning is raised and run find_optimal_sampen_params
    with pytest.warns(OptimizationFailureWarning):
        sampen.find_optimal_sampen_params()

    # Check that m_star_, r_star_, p_star_ are NaN (as it cannot find optimal values for short signals)
    assert np.isnan(sampen.m_star_)
    assert np.isnan(sampen.r_star_)
    assert np.isnan(sampen.p_star_)


def test_find_optimal_sampen_params_tight_r_range():
    rng = np.random.default_rng(17)
    signal_1 = rng.normal(size=100)
    signal_2 = rng.normal(size=100)

    df = pd.DataFrame(
        {
            "signal_id": np.concatenate(
                [np.repeat(0, len(signal_1)), np.repeat(1, len(signal_2))]
            ),
            "timestamp": np.concatenate([signal_1, signal_2]),
            "value": rng.normal(size=len(signal_1) + len(signal_2)),
        }
    )

    # Initialize SampleEntropy with a tight r_range
    sampen = SampleEntropy(
        df, SampEnSettings(n_trials=10, r_range=(0.02, 0.03), random_seed=17)
    )

    # Check that OptimizationFailureWarning is raised and run find_optimal_sampen_params
    with pytest.warns(OptimizationFailureWarning):
        sampen.find_optimal_sampen_params()

    # Check that m_star_, r_star_, p_star_ are NaN (as it cannot find optimal values with the tight r_range)
    assert np.isnan(sampen.m_star_)
    assert np.isnan(sampen.r_star_)
    assert np.isnan(sampen.p_star_)


def test_compute_all_sampen_no_params():
    rng = np.random.default_rng(17)
    df = pd.DataFrame(
        {
            "signal_id": np.repeat(range(5), 200),
            "timestamp": np.tile(range(200), 5),
            "value": rng.normal(size=1000),
        }
    )

    # Initialize SampleEntropy with no params
    sampen = SampleEntropy(df, SampEnSettings())

    with pytest.raises(ValueError):
        sampen.compute_all_sampen(optimize=False)


def test_compute_all_sampen_failed_optimization():
    rng = np.random.default_rng(17)
    df = pd.DataFrame(
        {
            "signal_id": np.repeat(range(5), 15),
            "timestamp": np.tile(range(15), 5),
            "value": rng.normal(size=(15 * 5)),
        }
    )

    # Initialize SampleEntropy with no params
    sampen = SampleEntropy(df, SampEnSettings(n_trials=10, random_seed=17))

    with pytest.warns(OptimizationFailureWarning):
        with pytest.raises(ValueError):
            sampen.compute_all_sampen(optimize=True)


def test_compute_all_sampen_nominal_case_with_optimization():
    rng = np.random.default_rng(17)
    df = pd.DataFrame(
        {
            "signal_id": np.repeat(range(5), 200),
            "timestamp": np.tile(range(200), 5),
            "value": rng.normal(size=1000),
        }
    )

    # Initialize SampleEntropy with params
    sampen = SampleEntropy(df, SampEnSettings(n_trials=10, random_seed=17))

    sampen_df = sampen.compute_all_sampen(optimize=True)

    assert len(sampen_df) == 5  # There are five unique signals
    assert (
        not sampen_df["sampen"].isna().any()
    )  # There are no NaN values in the SampEn estimates


def test_compute_all_sampen_nominal_case_no_optimization():
    rng = np.random.default_rng(17)
    df = pd.DataFrame(
        {
            "signal_id": np.repeat(range(5), 200),
            "timestamp": np.tile(range(200), 5),
            "value": rng.normal(size=1000),
        }
    )

    # Initialize SampleEntropy with params
    params = SampEnSettings(m=1, r=0.2)
    sampen = SampleEntropy(df, params)

    sampen_df = sampen.compute_all_sampen(optimize=False)

    assert len(sampen_df) == 5  # There are five unique signals
    assert (
        not sampen_df["sampen"].isna().any()
    )  # There are no NaN values in the SampEn estimates


def test_deterministic_optimization_output():
    rng = np.random.default_rng(17)
    df = pd.DataFrame(
        {
            "signal_id": np.repeat(range(5), 200),
            "timestamp": np.tile(range(200), 5),
            "value": rng.normal(size=1000),
        }
    )

    # Initialize SampleEntropy with params with known random seed
    sampen1 = SampleEntropy(df, SampEnSettings(n_trials=10, random_seed=17))
    sampen1.find_optimal_sampen_params()
    m_star1 = sampen1.m_star_
    r_star1 = sampen1.r_star_
    p_star1 = sampen1.p_star_

    # Repeat same prodedure to ensure we get the same optimization output
    sampen2 = SampleEntropy(df, SampEnSettings(n_trials=10, random_seed=17))
    sampen2.find_optimal_sampen_params()
    m_star2 = sampen2.m_star_
    r_star2 = sampen2.r_star_
    p_star2 = sampen2.p_star_

    # Should have the same optimization output
    assert m_star1 == m_star2
    assert r_star1 == r_star2
    assert p_star1 == p_star2


def test_find_optimal_sampen_params_diff_rng_seeds():
    rng = np.random.default_rng(17)
    values = rng.normal(size=1000)
    df = pd.DataFrame(
        {
            "signal_id": np.repeat(range(5), 200),
            "timestamp": np.tile(range(200), 5),
            "value": values,
        }
    )

    for seed in range(3):
        sampen = SampleEntropy(df, SampEnSettings(n_trials=10, m=1, random_seed=seed))
        sampen.find_optimal_sampen_params()
        assert sampen.m_star_ is not None
        assert sampen.r_star_ is not None
        assert sampen.p_star_ is not None
