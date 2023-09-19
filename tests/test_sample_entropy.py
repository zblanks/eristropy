import numpy as np
import pandas as pd
import pytest

from eristropy.sample_entropy import (
    OptimizationFailureWarning,
    SampEnSettingWarning,
    SampleEntropy,
)
from eristropy.utils import _seed


@pytest.fixture
def sample_df():
    signal_ids = np.repeat(["abc", "def"], 100)
    timestamps = np.tile(np.arange(100), 2)
    abc_values = np.linspace(0, 100, 100)
    def_values = np.sin(np.linspace(0, 2 * np.pi, 100))
    values = np.concatenate((abc_values, def_values))

    df = pd.DataFrame(
        {"signal_id": signal_ids, "timestamp": timestamps, "value": values}
    )
    return df


def test_check_ranges(sample_df):
    # Invalid r_range
    with pytest.raises(ValueError):
        SampleEntropy(sample_df, r_range=(0.50, 0.10))  # second value less than first
    with pytest.raises(ValueError):
        SampleEntropy(sample_df, r_range=(-0.10, 0.50))  # first value less than 0

    # Invalid m_range
    with pytest.raises(ValueError):
        SampleEntropy(sample_df, m_range=(3, 1))  # second value less than first
    with pytest.raises(ValueError):
        SampleEntropy(sample_df, m_range=(1.5, 3))  # first value not an integer
    with pytest.raises(ValueError):
        SampleEntropy(sample_df, m_range=(1, 3.5))  # second value not an integer

    # Invalid p_range
    with pytest.raises(ValueError):
        SampleEntropy(sample_df, p_range=(0.99, 0.01))  # second value less than first
    with pytest.raises(ValueError):
        SampleEntropy(sample_df, p_range=(-0.01, 0.99))  # first value less than 0
    with pytest.raises(ValueError):
        SampleEntropy(sample_df, p_range=(0.01, 1.1))  # second value greater than 1


def test_check_fixed_values(sample_df):
    # Invalid m
    with pytest.raises(ValueError):
        SampleEntropy(sample_df, m=-1)  # m less than 0
    with pytest.raises(ValueError):
        SampleEntropy(sample_df, m=0)  # m equals 0
    with pytest.raises(ValueError):
        SampleEntropy(sample_df, m=1.5)  # m not an integer

    # Invalid r
    with pytest.raises(ValueError):
        SampleEntropy(sample_df, r=-0.1)  # r less than 0
    with pytest.raises(ValueError):
        SampleEntropy(sample_df, r=0)  # r equals 0

    # Invalid p
    with pytest.raises(ValueError):
        SampleEntropy(sample_df, p=-0.1)  # p less than 0
    with pytest.raises(ValueError):
        SampleEntropy(sample_df, p=1.1)  # p greater than 1
    with pytest.raises(ValueError):
        SampleEntropy(sample_df, p=1)  # p equals 1
    with pytest.raises(ValueError):
        SampleEntropy(sample_df, p=0)  # p equals 0

    # Invalid lam
    with pytest.raises(ValueError):
        SampleEntropy(sample_df, lam=-0.1)  # lam less than 0


def test_check_positive_integer(sample_df):
    # Invalid n_boot
    with pytest.raises(ValueError):
        SampleEntropy(sample_df, n_boot=-1)  # n_boot less than 0
    with pytest.raises(ValueError):
        SampleEntropy(sample_df, n_boot=0)  # n_boot equals 0
    with pytest.raises(ValueError):
        SampleEntropy(sample_df, n_boot=1.5)  # n_boot not an integer

    # Invalid n_trials
    with pytest.raises(ValueError):
        SampleEntropy(sample_df, n_trials=-1)  # n_trials less than 0
    with pytest.raises(ValueError):
        SampleEntropy(sample_df, n_trials=0)  # n_trials equals 0
    with pytest.raises(ValueError):
        SampleEntropy(sample_df, n_trials=1.5)  # n_trials not an integer

    # Invalid random_seed
    with pytest.raises(ValueError):
        SampleEntropy(sample_df, random_seed=-1)  # random_seed less than 0
    with pytest.raises(ValueError):
        SampleEntropy(sample_df, random_seed=1.5)  # random_seed not an integer


def test_check_string_attrs(sample_df):
    # Invalid signal_id
    with pytest.raises(ValueError):
        SampleEntropy(sample_df, signal_id=123)  # signal_id is not a string

    # Invalid timestamp
    with pytest.raises(ValueError):
        SampleEntropy(sample_df, timestamp=123)  # timestamp is not a string

    # Invalid value_col
    with pytest.raises(ValueError):
        SampleEntropy(sample_df, value_col=123)  # value_col is not a string


def test_default_values(sample_df):
    sampen = SampleEntropy(sample_df)

    assert sampen.signal_id == "signal_id"
    assert sampen.timestamp == "timestamp"
    assert sampen.value_col == "value"
    assert sampen.n_boot == 100
    assert sampen.n_trials == 100
    assert sampen.random_seed is None
    assert sampen.r_range == (0.10, 0.50)
    assert sampen.m_range == (1, 3)
    assert sampen.p_range == (0.01, 0.99)
    assert sampen.lam == 0.33
    assert sampen.r is None
    assert sampen.m is None
    assert sampen.p is None


def test_custom_values(sample_df):
    sample_df = sample_df.rename(
        columns={
            "signal_id": "my_signal_id",
            "timestamp": "my_timestamp",
            "value": "my_value",
        }
    )

    sampen = SampleEntropy(
        sample_df,
        signal_id="my_signal_id",
        timestamp="my_timestamp",
        value_col="my_value",
        n_boot=200,
        n_trials=300,
        random_seed=42,
        r_range=(0.15, 0.45),
        m_range=(2, 4),
        p_range=(0.05, 0.95),
        lam=0.25,
        r=0.2,
        m=2,
        p=0.1,
    )

    assert sampen.signal_id == "my_signal_id"
    assert sampen.timestamp == "my_timestamp"
    assert sampen.value_col == "my_value"
    assert sampen.n_boot == 200
    assert sampen.n_trials == 300
    assert sampen.random_seed == 42
    assert sampen.r_range == (0.15, 0.45)
    assert sampen.m_range == (2, 4)
    assert sampen.p_range == (0.05, 0.95)
    assert sampen.lam == 0.25
    assert sampen.r == 0.2
    assert sampen.m == 2
    assert sampen.p == 0.1


def test_warns_on_low_values(sample_df):
    # Low n_boot
    with pytest.warns(SampEnSettingWarning):
        SampleEntropy(sample_df, n_boot=1)

    # Low n_trials
    with pytest.warns(SampEnSettingWarning):
        SampleEntropy(sample_df, n_trials=1)


def test_warns_on_extreme_fixed_values(sample_df):
    # Extreme r
    with pytest.warns(SampEnSettingWarning):
        SampleEntropy(sample_df, r=0.50)

    # Extreme p
    with pytest.warns(SampEnSettingWarning):
        SampleEntropy(sample_df, p=0.01)


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
    _seed(17)  # Deterministic test output
    mse1 = SampleEntropy._bootstrap_mse(x1, m, r, p, n_boot)
    mse2 = SampleEntropy._bootstrap_mse(x2, m, r, p, n_boot)
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

    sampen = SampleEntropy(df, n_trials=10, random_seed=17)
    assert isinstance(sampen, SampleEntropy)
    assert sampen.df.equals(df)


def test_sample_entropy_init_optimization_vars():
    rng = np.random.default_rng(17)
    df = pd.DataFrame(
        {
            "signal_id": np.repeat(range(5), 200),
            "timestamp": np.tile(range(200), 5),
            "value": rng.normal(size=1000),
        }
    )
    sampen = SampleEntropy(df, m=2, r=0.2, p=0.5)
    assert sampen.m_star_ == sampen.m
    assert sampen.r_star_ == sampen.r
    assert sampen.p_star_ == sampen.p


def test_sample_entropy_init_signal_groups():
    rng = np.random.default_rng(17)
    df = pd.DataFrame(
        {
            "signal_id": np.repeat(range(5), 200),
            "timestamp": np.tile(range(200), 5),
            "value": rng.normal(size=1000),
        }
    )
    sampen = SampleEntropy(df)
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
    with pytest.raises(
        ValueError,
        match=(
            "The upper limit of m_range cannot exceed the length "
            "of the shortest time series in the DataFrame."
        ),
    ):
        SampleEntropy(df, m_range=(1, 250))


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
    sampen = SampleEntropy(df, n_trials=10, random_seed=17)

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
    assert sampen.m_range[0] <= sampen.m_star_ <= sampen.m_range[1]
    assert sampen.r_range[0] <= sampen.r_star_ <= sampen.r_range[1]
    assert sampen.p_range[0] <= sampen.p_star_ <= sampen.p_range[1]


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
    sampen = SampleEntropy(df, n_trials=10, random_seed=17)

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
    assert sampen.m_range[0] <= sampen.m_star_ <= sampen.m_range[1]
    assert sampen.r_range[0] <= sampen.r_star_ <= sampen.r_range[1]
    assert sampen.p_range[0] <= sampen.p_star_ <= sampen.p_range[1]


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
    sampen = SampleEntropy(df, n_trials=10, random_seed=17)

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
    sampen = SampleEntropy(df, n_trials=10, r_range=(0.02, 0.03), random_seed=17)

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
    sampen = SampleEntropy(df)

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
    sampen = SampleEntropy(df, n_trials=10, random_seed=17)

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
    sampen = SampleEntropy(df, n_trials=10, random_seed=17)
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

    sampen = SampleEntropy(df, m=1, r=0.2)
    sampen_df = sampen.compute_all_sampen(optimize=False)

    assert len(sampen_df) == 5  # There are five unique signals
    assert (
        not sampen_df["sampen"].isna().any()
    )  # There are no NaN values in the SampEn estimates


def test_compute_all_sampen_nominal_case_no_optimization_with_uncertainty():
    rng = np.random.default_rng(17)
    df = pd.DataFrame(
        {
            "signal_id": np.repeat(range(5), 200),
            "timestamp": np.tile(range(200), 5),
            "value": rng.normal(size=1000),
        }
    )

    sampen = SampleEntropy(df, m=1, r=0.2, p=0.5)
    sampen_df = sampen.compute_all_sampen(optimize=False, estimate_uncertainty=True)

    assert len(sampen_df) == 5  # There are five unique signals
    assert (
        not sampen_df["sampen"].isna().any()
    )  # There are no NaN values in the SampEn estimates

    # SE(SampEn) must be non-negative
    assert (sampen_df["se_sampen"] >= 0).all()


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
    sampen1 = SampleEntropy(df, n_trials=10, random_seed=17)
    sampen1.find_optimal_sampen_params()
    m_star1 = sampen1.m_star_
    r_star1 = sampen1.r_star_
    p_star1 = sampen1.p_star_

    # Repeat same prodedure to ensure we get the same optimization output
    sampen2 = SampleEntropy(df, n_trials=10, random_seed=17)
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
        sampen = SampleEntropy(df, n_trials=10, m=1, random_seed=seed)
        sampen.find_optimal_sampen_params()
        assert sampen.m_star_ is not None
        assert sampen.r_star_ is not None
        assert sampen.p_star_ is not None
