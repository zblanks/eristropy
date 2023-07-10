import math
import numpy as np

from cpyet.sample_entropy import _sampen, _cp_mean_and_sd


def test_sampen():
    rng = np.random.default_rng(17)
    x = rng.random(100)
    m = 2
    r = 0.2 * x.std()

    result = _sampen(x, m, r)

    # Assume _cp_mean_and_sd is valid -- derived from Lake et al. implementation
    # in C
    expected_result = -math.log(_cp_mean_and_sd(x, m, r)[0])
    np.testing.assert_almost_equal(result, expected_result)
