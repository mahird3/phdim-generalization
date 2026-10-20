import numpy as np
import pytest

from phdim import distance_matrix, ph0_lifetimes, ph_dim
from phdim.shapes import circle, cube, embed, sierpinski


# ---------------------------------------------------------------- distance_matrix and ph0_lifetimes

def test_distance_matrix_known_values():
    D = distance_matrix([[0, 0], [3, 0], [0, 4]])
    np.testing.assert_allclose(D, [[0, 3, 4], [3, 0, 5], [4, 5, 0]])


def test_lifetimes_of_points_on_a_line():
    # gaps 1, 2, 3: the MST joins neighbours, so the lifetimes are the gaps
    D = distance_matrix([[0.0], [1.0], [3.0], [6.0]])
    np.testing.assert_allclose(np.sort(ph0_lifetimes(D)), [1, 2, 3])


@pytest.mark.parametrize("k", [2, 10, 100])
def test_tree_on_k_points_has_k_minus_1_edges(k):
    D = distance_matrix(np.random.default_rng(k).random((k, 4)))
    assert ph0_lifetimes(D).size == k - 1


def test_lifetimes_scale_with_the_points():
    X = np.random.default_rng(0).random((100, 3))
    np.testing.assert_allclose(np.sort(ph0_lifetimes(distance_matrix(10 * X))),
                               10 * np.sort(ph0_lifetimes(distance_matrix(X))))


def test_duplicate_points_raise():
    X = np.random.default_rng(0).random((10, 2))
    X[3] = X[7]
    with pytest.raises(ValueError, match="duplicate"):
        ph0_lifetimes(distance_matrix(X))


def test_lifetimes_match_ripser():
    """The MST edge lengths are exactly the finite deaths of the H0 persistence diagram."""
    ripser = pytest.importorskip("ripser").ripser
    P = np.random.default_rng(0).random((200, 5))
    h0 = ripser(P, maxdim=0)["dgms"][0]
    deaths = np.sort(h0[np.isfinite(h0[:, 1]), 1])
    np.testing.assert_allclose(deaths, np.sort(ph0_lifetimes(distance_matrix(P))), atol=1e-5)


# ---------------------------------------------------------------- ph_dim on sets of known dimension

def _square_in_R100(n, rng):
    return embed(cube(n, 2, rng), 100, rng)


# name: (sampler, true dimension, tolerance). Filled sets come out a little low at n = 1000
# (points near the boundary have longer MST edges), hence the wider tolerances there.
KNOWN_SETS = {
    "circle": (circle, 1.0, 0.10),
    "square": (lambda n, rng: cube(n, 2, rng), 2.0, 0.20),
    "square in R^100": (_square_in_R100, 2.0, 0.20),
    "cube": (lambda n, rng: cube(n, 3, rng), 3.0, 0.35),
    "Sierpinski": (sierpinski, np.log(3) / np.log(2), 0.15),
}


@pytest.mark.parametrize("name", KNOWN_SETS)
def test_ph_dim_recovers_known_dimension(name):
    sampler, truth, tol = KNOWN_SETS[name]
    X = sampler(1000, np.random.default_rng(0))
    assert abs(ph_dim(X, n_repeats=3) - truth) < tol


def test_ph_dim_same_from_points_or_distances():
    X = cube(300, 2, np.random.default_rng(0))
    assert ph_dim(W=X, seed=1) == ph_dim(D=distance_matrix(X), seed=1)


def test_ph_dim_scale_invariant():
    """Scaling the points shifts log E by a constant, so the slope and the dimension do not change."""
    X = cube(300, 2, np.random.default_rng(0))
    np.testing.assert_allclose(ph_dim(1000 * X), ph_dim(X), rtol=1e-9)
