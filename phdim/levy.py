"""Isotropic beta-stable Levy paths: point clouds whose dimension is known (it equals beta).

Definition used in the paper's appendix: independent, stationary increments with
E exp(i<w, L_1>) = exp(-|w|^beta), 0 < beta <= 2. beta = 2 is Brownian motion.
"""

import numpy as np


def levy_increments(n, d, beta, rng):
    """n i.i.d. isotropic beta-stable vectors in R^d with E exp(i<w, X>) = exp(-|w|^beta).

    X = sqrt(2A) * G with G ~ N(0, I_d) and A > 0 a (beta/2)-stable variable with
    E exp(-sA) = exp(-s^(beta/2)), drawn with Kanter's formula. beta = 2 gives A = 1 (Gaussian).
    """
    if not 0 < beta <= 2:
        raise ValueError(f"beta must be in (0, 2], got {beta}")
    a = beta / 2
    if a < 1:
        U, E = rng.random(n), rng.exponential(size=n)
        A = np.sin(a * np.pi * U) / np.sin(np.pi * U) ** (1 / a) * (np.sin((1 - a) * np.pi * U) / E) ** ((1 - a) / a)
    else:
        A = np.ones(n)
    return np.sqrt(2 * A)[:, None] * rng.standard_normal((n, d))


def levy_path(K, d, beta, rng):
    """K points of a beta-stable Levy path in R^d started at 0 (unit time steps)."""
    return np.vstack([np.zeros((1, d)), np.cumsum(levy_increments(K - 1, d, beta, rng), axis=0)])
