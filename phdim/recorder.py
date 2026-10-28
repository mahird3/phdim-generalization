"""Record the iterates of any training loop, to compute their PH dimension.

The recorder saves the flattened weights of a model after each optimizer step, once started:

    rec = Recorder(net, n_iterates=200)
    rec.attach(opt)                      # records after every opt.step(), the loop stays unchanged
    ...                                  # train; call rec.start() when near a minimum
    rec.ph_dim()                         # or ph_dim(D=rec.distance_matrix()) to reuse D

record() does the same by hand, and the decorator @rec.track records after each call of a step function.

For large models, proj_dim=k stores a random projection P w (k numbers) instead of the weights w.
By the Johnson-Lindenstrauss lemma all distances between iterates are kept up to about 1/sqrt(2k),
whatever the number of weights, so the PH dimension is nearly unchanged.
"""

import functools

import torch
from torch.nn.utils import parameters_to_vector

from .estimator import distance_matrix, ph_dim


class Recorder:
    """Flattened weights of `model` after each step, n_iterates of them, starting at start().

    every=k keeps one step out of k. The iterates are stored on the model's device, in its dtype.
    proj_dim=k stores P (w - w_0) instead, with P a fixed (k, number of weights) Gaussian matrix drawn from `seed`
    and w_0 the first recorded iterate. Subtracting w_0 first keeps the small distances exact in float32
    (P w would be as large as w, and its rounding errors as large as the steps).
    """

    def __init__(self, model, n_iterates=200, every=1, proj_dim=None, seed=0):
        self.model, self.n_iterates, self.every = model, n_iterates, every
        self.proj_dim, self.seed = proj_dim, seed
        self.active, self.steps, self.n = False, 0, 0
        self._traj, self._hook, self._w0 = None, None, None

    def start(self):
        self.active = True

    @property
    def done(self):
        return self.n >= self.n_iterates

    def record(self):
        """Save the current weights (if started, not done, and on a kept step)."""
        if not self.active or self.done:
            return
        self.steps += 1
        if self.steps % self.every:
            return
        w = parameters_to_vector(self.model.parameters()).detach()
        if self.proj_dim is not None:
            if self._w0 is None:
                self._w0 = w.clone()
            w = self.project(w - self._w0)
        if self._traj is None:
            self._traj = torch.empty(self.n_iterates, len(w), dtype=w.dtype, device=w.device)
        self._traj[self.n] = w
        self.n += 1

    @torch.no_grad()
    def project(self, w, block_size=2**24):
        """P w, with P (proj_dim, len(w)) of N(0, 1/proj_dim) entries.

        P is never stored: it is drawn again from the seed at each call, block_size entries at a time
        (64 MB in float32), so it is the same matrix for every iterate.
        """
        k = self.proj_dim
        cols = max(1, block_size // k)
        gen = torch.Generator(device=w.device).manual_seed(self.seed)
        y = torch.zeros(k, dtype=w.dtype, device=w.device)
        for c in range(0, len(w), cols):
            P_block = torch.randn(k, len(w[c:c + cols]), generator=gen, dtype=w.dtype, device=w.device)
            y += P_block @ w[c:c + cols]
        return y / k ** 0.5

    def attach(self, optimizer):
        """Record after every optimizer.step(), so once per update even with gradient accumulation."""
        self._hook = optimizer.register_step_post_hook(lambda opt, args, kwargs: self.record())
        return self

    def detach(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def track(self, step_fn):
        """Decorator: record after each call of step_fn."""
        @functools.wraps(step_fn)
        def wrapper(*args, **kwargs):
            out = step_fn(*args, **kwargs)
            self.record()
            return out
        return wrapper

    @property
    def traj(self):
        """(iterates so far, number of weights or proj_dim) tensor."""
        return self._traj[:self.n] if self._traj is not None else torch.empty(0, 0)

    def distance_matrix(self):
        return distance_matrix(self.traj)

    def ph_dim(self, **kwargs):
        """PH dimension of the recorded iterates; kwargs (alpha, n_repeats, ...) go to phdim.ph_dim."""
        return ph_dim(D=self.distance_matrix(), **kwargs)
