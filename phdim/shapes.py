"""Point clouds with a known dimension, used to validate the estimator."""

import numpy as np


def circle(n, rng):
    """n points on the unit circle (dimension 1)."""
    t = rng.uniform(0, 2 * np.pi, n)
    return np.c_[np.cos(t), np.sin(t)]


def cube(n, d, rng):
    """n uniform points in [0, 1]^d (dimension d)."""
    return rng.random((n, d))


def sierpinski(n, rng, burn_in=50):
    """Chaos game: n points on the Sierpinski triangle (dimension log 3 / log 2)."""
    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    p, out = np.array([0.3, 0.3]), np.empty((n + burn_in, 2))
    for i in range(n + burn_in):
        p = (p + V[rng.integers(3)]) / 2
        out[i] = p
    return out[burn_in:]


def embed(X, p, rng):
    """Map the points X (n, d) linearly into R^p with a random Gaussian matrix (dimension unchanged)."""
    return X @ rng.standard_normal((X.shape[1], p))
