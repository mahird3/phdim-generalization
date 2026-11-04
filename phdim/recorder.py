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

loss_on=(X, y) stores the per-example losses of the model on these examples instead of the weights,
and the distance becomes the data-dependent rho_S of Dupuis, Deligiannidis, Simsekli (ICML 2023):
the mean absolute change of the losses. It ignores weight changes that do not change the losses
(e.g. ReLU rescalings) and needs no Lipschitz constant. A random subset of the training set is enough
(Fig. 3 of their paper): n_examples=m picks one. It costs one forward pass over these examples per recorded iterate.
"""

import functools

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector

from .estimator import distance_matrix, loss_distance_matrix, ph_dim


class Recorder:
    """Flattened weights of `model` after each step, n_iterates of them, starting at start().

    every=k keeps one step out of k. The iterates are stored on the model's device, in its dtype.
    proj_dim=k stores P (w - w_0) instead, with P a fixed (k, number of weights) Gaussian matrix drawn from `seed`
    and w_0 the first recorded iterate. Subtracting w_0 first keeps the small distances exact in float32
    (P w would be as large as w, and its rounding errors as large as the steps).
    loss_on=(X, y), tensors on the model's device, stores loss_fn(model(X), y) instead: one loss per example,
    computed in eval mode (dropout off, batch norm statistics frozen). loss_fn defaults to the cross-entropy;
    any loss with reduction="none" works, e.g. loss_fn=nn.MSELoss(reduction="none") for regression with one output.
    n_examples=m keeps a random subset of m examples of loss_on, drawn once from `seed`, the same at every step.
    """

    def __init__(self, model, n_iterates=200, every=1, proj_dim=None, seed=0, loss_on=None, loss_fn=None,
                 n_examples=None):
        if proj_dim is not None and loss_on is not None:
            raise ValueError("proj_dim and loss_on record different things: use one recorder for each")
        if n_examples is not None:
            if loss_on is None or n_examples > len(loss_on[0]):
                raise ValueError("n_examples needs loss_on with at least n_examples examples")
            X, y = loss_on
            idx = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed))[:n_examples].to(X.device)
            loss_on = (X[idx], y[idx])
        self.model, self.n_iterates, self.every = model, n_iterates, every
        self.proj_dim, self.seed = proj_dim, seed
        self.loss_on = loss_on
        self.loss_fn = loss_fn or nn.CrossEntropyLoss(reduction="none")
        self.active, self.steps, self.n = False, 0, 0
        self._traj, self._hook, self._w0 = None, None, None

    def start(self):
        self.active = True

    @property
    def done(self):
        return self.n >= self.n_iterates

    def record(self):
        """Save the current weights, or losses (if started, not done, and on a kept step)."""
        if not self.active or self.done:
            return
        self.steps += 1
        if self.steps % self.every:
            return
        if self.loss_on is not None:
            w = self.losses()
        else:
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

    @torch.no_grad()
    def losses(self, chunk=10_000):
        """Per-example losses of the current model on loss_on, in eval mode, chunk examples at a time."""
        X, y = self.loss_on
        was_training = self.model.training
        self.model.eval()
        try:
            out = torch.cat([self.loss_fn(self.model(X[i:i + chunk]), y[i:i + chunk]).reshape(-1)
                             for i in range(0, len(X), chunk)])
        finally:
            self.model.train(was_training)
        if out.shape != (len(X),):
            raise ValueError("loss_fn must return one loss per example (reduction='none')")
        return out

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
        """(iterates so far, number of weights, proj_dim or number of loss_on examples) tensor."""
        return self._traj[:self.n] if self._traj is not None else torch.empty(0, 0)

    def distance_matrix(self):
        """Distances between the recorded iterates: Euclidean, or rho_S with loss_on."""
        if self.loss_on is not None:
            return loss_distance_matrix(self.traj)
        return distance_matrix(self.traj)

    def ph_dim(self, **kwargs):
        """PH dimension of the recorded iterates; kwargs (alpha, n_repeats, ...) go to phdim.ph_dim."""
        return ph_dim(D=self.distance_matrix(), **kwargs)
