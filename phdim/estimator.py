"""PH0 dimension estimator: Algorithm 1 of Birdal et al. (2021).

The finite PH0 lifetimes of a point cloud are the edge lengths of its minimum
spanning tree (MST). For k points sampled from a set of dimension D, the
alpha-weighted total MST length grows like

    E_alpha(k) = sum_e |e|^alpha  ~  k^(1 - alpha / D),

so the slope m of log E_alpha(k) against log k gives D = alpha / (1 - m).
"""

import numpy as np
import torch
from scipy.sparse.csgraph import minimum_spanning_tree


def distance_matrix(W):
    """(K, p) array or tensor, one point (e.g. one flattened iterate) per row -> (K, K) Euclidean distances.

    The distances are computed on the device of W (a GPU tensor stays on the GPU, a numpy array
    runs on the CPU), and only the small (K, K) result is returned, as a numpy array.
    """
    W = torch.as_tensor(W, dtype=torch.float64)
    W = W - W.mean(dim=0)  # remove the offset shared by all points (precision only)
    # direct formula: |a|^2 + |b|^2 - 2ab would cancel out the small distances between nearby iterates
    return torch.cdist(W, W, compute_mode="donot_use_mm_for_euclid_dist").cpu().numpy()


def loss_distance_matrix(L):
    """(K, n) per-example losses, one iterate per row -> (K, K) data-dependent distances

        rho_S(w, w') = (1/n) sum_i |loss(w, z_i) - loss(w', z_i)|

    of Dupuis, Deligiannidis, Simsekli (ICML 2023). Like distance_matrix, it runs on the device of L
    and returns a numpy array. It is a pseudo-metric: two different weight vectors with the same losses
    on every example (e.g. a ReLU rescaling of each other) are at distance 0.
    """
    L = torch.as_tensor(L, dtype=torch.float64)
    return (torch.cdist(L, L, p=1) / L.shape[1]).cpu().numpy()


def ph0_lifetimes(D):
    """(k, k) distance matrix -> the k-1 finite PH0 lifetimes, i.e. the MST edge lengths.

    Zero distances (duplicate points, or a pseudo-metric such as rho_S) give lifetimes 0.
    """
    zero = D == 0
    np.fill_diagonal(zero, False)
    if not zero.any():
        return minimum_spanning_tree(D).data
    # scipy reads a 0 entry as "no edge" and would use a longer edge instead:
    # make the zero edges the shortest possible ones, then set them back to 0
    tiny = np.finfo(np.float64).tiny
    lifetimes = minimum_spanning_tree(np.where(zero, tiny, D)).data
    lifetimes[lifetimes == tiny] = 0
    return lifetimes


def ph_dim(W=None, D=None, alpha=1.0, k_min=None, k_step=None, n_repeats=1, seed=0, return_curve=False):
    """Algorithm 1: PH0 dimension estimate alpha / (1 - m).

    Pass the points W (K, p), as a numpy array or a torch tensor, or a precomputed distance matrix D (K, K).
    m is the least-squares slope of log E_alpha(k) against log k, where
    E_alpha(k) is the alpha-weighted MST length of a random subset of size k.
    By default about 15 subset sizes are used, from K/10 (at least 20) up to K.
    alpha must be smaller than the true dimension, otherwise the estimate tends to alpha.
    """
    if D is None:
        D = distance_matrix(W)
    K = D.shape[0]
    k_min = k_min or max(20, K // 10)
    k_step = k_step or max(1, (K - k_min) // 14)
    ks = np.arange(k_min, K + 1, k_step)
    rng = np.random.default_rng(seed)

    log_E = np.empty(len(ks))
    for i, k in enumerate(ks):
        reps = []
        for _ in range(n_repeats):
            idx = rng.choice(K, size=k, replace=False)
            reps.append(np.log(np.sum(ph0_lifetimes(D[np.ix_(idx, idx)]) ** alpha)))
        log_E[i] = np.mean(reps)

    m, b = np.polyfit(np.log(ks), log_E, 1)
    dim = alpha / (1 - m)
    if return_curve:
        return dim, {"k": ks, "log_E": log_E, "slope": m, "intercept": b}
    return dim
