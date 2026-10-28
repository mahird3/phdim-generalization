import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector

from phdim import distance_matrix, ph_dim
from phdim.recorder import Recorder


def setup(seed=0):
    """A small network, its optimizer and data, as in any user's training script."""
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 3))
    opt = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    X = torch.randn(64, 10)
    return net, opt, X, X[:, :3].argmax(1)


def plain_loop(net, opt, X, y, n_steps, rec=None, start_at=0):
    """The usual inline loop; the recorder only needs to be started."""
    weights = []
    for step in range(n_steps):
        if rec is not None and step == start_at:
            rec.start()
        i = (8 * step) % 64
        opt.zero_grad()
        F.cross_entropy(net(X[i:i + 8]), y[i:i + 8]).backward()
        opt.step()
        weights.append(parameters_to_vector(net.parameters()).detach().clone())
    return torch.stack(weights)


def test_attach_records_after_each_step_once_started():
    net, opt, X, y = setup()
    rec = Recorder(net, n_iterates=5).attach(opt)
    weights = plain_loop(net, opt, X, y, n_steps=20, rec=rec, start_at=10)
    assert rec.done
    torch.testing.assert_close(rec.traj, weights[10:15], rtol=0, atol=0)
    np.testing.assert_allclose(rec.distance_matrix(), distance_matrix(weights[10:15]))


def test_recorder_does_not_change_training():
    net, opt, X, y = setup()
    without = plain_loop(net, opt, X, y, n_steps=20)
    net, opt, X, y = setup()
    rec = Recorder(net, n_iterates=10).attach(opt)
    with_rec = plain_loop(net, opt, X, y, n_steps=20, rec=rec)
    torch.testing.assert_close(with_rec, without, rtol=0, atol=0)


def test_gradient_accumulation_records_once_per_update():
    net, opt, X, y = setup()
    rec = Recorder(net, n_iterates=100).attach(opt)
    rec.start()
    for step in range(6):
        opt.zero_grad()
        for i in range(0, 32, 8):  # 4 small batches per update
            F.cross_entropy(net(X[i:i + 8]), y[i:i + 8]).backward()
        opt.step()
    assert rec.n == 6


def test_every_and_detach():
    net, opt, X, y = setup()
    rec = Recorder(net, n_iterates=100, every=3).attach(opt)
    weights = plain_loop(net, opt, X, y, n_steps=12, rec=rec)
    torch.testing.assert_close(rec.traj, weights[2::3], rtol=0, atol=0)
    rec.detach()
    plain_loop(net, opt, X, y, n_steps=5)
    assert rec.n == 4


def test_track_decorator():
    net, opt, X, y = setup()
    rec = Recorder(net, n_iterates=3)

    @rec.track
    def train_step(Xb, yb):
        opt.zero_grad()
        loss = F.cross_entropy(net(Xb), yb)
        loss.backward()
        opt.step()
        return loss.item()

    rec.start()
    losses = [train_step(X[i:i + 8], y[i:i + 8]) for i in range(0, 40, 8)]
    assert len(losses) == 5 and isinstance(losses[0], float)  # the step still returns its value
    assert rec.done and rec.traj.shape == (3, sum(p.numel() for p in net.parameters()))


def test_ph_dim_shortcut():
    net, opt, X, y = setup()
    rec = Recorder(net, n_iterates=100).attach(opt)
    plain_loop(net, opt, X, y, n_steps=100, rec=rec)
    assert rec.ph_dim(alpha=0.5) == ph_dim(D=rec.distance_matrix(), alpha=0.5)
