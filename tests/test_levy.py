import numpy as np
import pytest

from phdim import distance_matrix, levy_increments, levy_path, ph_dim


# ---------------------------------------------------------------- the sampler

def test_path_starts_at_zero_and_sums_increments():
    P = levy_path(100, 7, 1.5, np.random.default_rng(0))
    assert P.shape == (100, 7)
    np.testing.assert_array_equal(P[0], 0)
    # the same seed gives the same increments
    steps = levy_increments(99, 7, 1.5, np.random.default_rng(0))
    np.testing.assert_allclose(np.diff(P, axis=0), steps, atol=1e-8)


@pytest.mark.parametrize("beta", [1.0, 1.5, 2.0])
def test_characteristic_function_in_a_random_direction(beta):
    """E cos(<w, X>) = exp(-|w|^beta) for every w: checks the law and the isotropy at once."""
    rng = np.random.default_rng(1)
    X = levy_increments(100_000, 3, beta, rng)
    u = rng.standard_normal(3)
    u /= np.linalg.norm(u)
    for r in [0.5, 1.0, 2.0]:
        emp = np.cos(r * X @ u).mean()
        assert abs(emp - np.exp(-r ** beta)) < 0.01


def test_beta_1_is_cauchy():
    """For beta = 1 each coordinate is standard Cauchy: P(|X| > 10) = 1 - (2/pi) arctan(10)."""
    X = levy_increments(200_000, 1, 1.0, np.random.default_rng(2))[:, 0]
    assert abs(np.mean(np.abs(X) > 10) - (1 - 2 / np.pi * np.arctan(10))) < 0.003


def test_invalid_beta_raises():
    with pytest.raises(ValueError):
        levy_increments(10, 2, 2.5, np.random.default_rng(0))


# ---------------------------------------------------------------- integration: the estimator on Levy paths
# Slower (a few seconds each): the whole pipeline on paths in R^128 whose dimension is beta.

def _mean_dim(beta, alpha=1.0, n_paths=3, K=1000, d=128, seed=0):
    rng = np.random.default_rng(seed)
    return np.mean([ph_dim(D=distance_matrix(levy_path(K, d, beta, rng)), alpha=alpha, seed=j)
                    for j in range(n_paths)])


def test_ph_dim_follows_tail_index():
    """The estimate tracks beta (Fig. 6 of the paper), slightly high."""
    betas = [1.2, 1.6, 2.0]
    est = [_mean_dim(b) for b in betas]
    assert est[0] < est[1] < est[2]
    for b, e in zip(betas, est):
        assert abs(e - b) < 0.35


def test_alpha_above_dimension_saturates():
    """With alpha > dimension the slope goes to 0 and the estimate just returns alpha (Fig. 7)."""
    assert abs(_mean_dim(1.2, alpha=2.0) - 2.0) < 0.15
