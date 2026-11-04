import numpy as np
import pytest
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


def test_projection_keeps_distances():
    net, opt, X, y = setup()
    full = Recorder(net, n_iterates=50).attach(opt)
    proj = Recorder(net, n_iterates=50, proj_dim=2000).attach(opt)  # both record the same iterates
    full.start()
    proj.start()
    plain_loop(net, opt, X, y, n_steps=50)
    assert proj.traj.shape == (50, 2000)
    iu = np.triu_indices(50, 1)
    ratio = proj.distance_matrix()[iu] / full.distance_matrix()[iu]
    assert np.abs(ratio - 1).max() < 0.1  # typical error 1/sqrt(2k) = 1.6%


def test_projection_is_one_fixed_linear_map():
    net = setup()[0]
    rec = Recorder(net, proj_dim=64, seed=3)
    g = torch.Generator().manual_seed(0)
    w1, w2 = torch.randn(1000, generator=g), torch.randn(1000, generator=g)
    for block_size in (2**24, 64 * 100):  # one block, or 10 blocks of 100 columns
        y1 = rec.project(w1, block_size)
        torch.testing.assert_close(rec.project(w1, block_size), y1, rtol=0, atol=0)  # same P at every call
        torch.testing.assert_close(rec.project(w1 + 2 * w2, block_size), y1 + 2 * rec.project(w2, block_size))


def test_projection_with_large_weights():
    # steps of 1e-3 on weights of size 1e4: float32 rounding of P w alone would be about as large as the steps
    net = setup()[0]
    with torch.no_grad():
        for p in net.parameters():
            p.fill_(1e4)
    full = Recorder(net, n_iterates=30)
    proj = Recorder(net, n_iterates=30, proj_dim=2000)
    full.start()
    proj.start()
    g = torch.Generator().manual_seed(0)
    for _ in range(30):
        with torch.no_grad():
            for p in net.parameters():
                p.add_(1e-3 * torch.randn(p.shape, generator=g))
        full.record()
        proj.record()
    iu = np.triu_indices(30, 1)
    ratio = proj.distance_matrix()[iu] / full.distance_matrix()[iu]
    assert np.abs(ratio - 1).max() < 0.1


def test_loss_recorder_stores_per_example_losses():
    net, opt, X, y = setup()
    rec = Recorder(net, n_iterates=5, loss_on=(X, y)).attach(opt)
    rec.start()
    expected = []
    for step in range(5):
        i = 8 * step
        opt.zero_grad()
        F.cross_entropy(net(X[i:i + 8]), y[i:i + 8]).backward()
        opt.step()
        with torch.no_grad():
            expected.append(F.cross_entropy(net(X), y, reduction="none"))
    expected = torch.stack(expected)
    torch.testing.assert_close(rec.traj, expected)
    L = expected.double()
    rho = (L[:, None, :] - L[None, :, :]).abs().mean(dim=2)  # (1/n) sum_i |l(w, z_i) - l(w', z_i)|
    np.testing.assert_allclose(rec.distance_matrix(), rho.numpy(), rtol=1e-6, atol=1e-12)


def test_loss_recorder_uses_eval_mode_and_does_not_change_training():
    def setup_with_dropout():
        net, _, X, y = setup()
        net = nn.Sequential(net[0], net[1], nn.Dropout(0.5), net[2])
        return net, torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9), X, y

    net, opt, X, y = setup_with_dropout()
    without = plain_loop(net, opt, X, y, n_steps=10)
    net, opt, X, y = setup_with_dropout()
    rec = Recorder(net, n_iterates=10, loss_on=(X, y)).attach(opt)
    with_rec = plain_loop(net, opt, X, y, n_steps=10, rec=rec)
    torch.testing.assert_close(with_rec, without, rtol=0, atol=0)  # the eval-mode forward uses no random numbers
    assert net.training  # training mode is restored after each recording
    net.eval()
    with torch.no_grad():
        torch.testing.assert_close(rec.traj[-1], F.cross_entropy(net(X), y, reduction="none"))  # dropout was off


def test_loss_distance_ignores_relu_rescaling():
    net, opt, X, y = setup()
    rec_w = Recorder(net, n_iterates=2)
    rec_l = Recorder(net, n_iterates=2, loss_on=(X, y))
    rec_w.start()
    rec_l.start()
    rec_w.record()
    rec_l.record()
    with torch.no_grad():  # hidden neuron j: incoming weights and bias times c_j, outgoing weights divided by c_j
        c = torch.linspace(0.5, 4, 16)
        net[0].weight.mul_(c[:, None])
        net[0].bias.mul_(c)
        net[2].weight.div_(c)
    rec_w.record()
    rec_l.record()
    assert rec_w.distance_matrix()[0, 1] > 0.5  # the weights moved
    assert rec_l.distance_matrix()[0, 1] < 1e-5  # the function, hence the losses, did not


def test_loss_recorder_checks_its_arguments():
    net, opt, X, y = setup()
    with pytest.raises(ValueError, match="one recorder for each"):
        Recorder(net, proj_dim=10, loss_on=(X, y))
    rec = Recorder(net, loss_on=(X, y), loss_fn=F.cross_entropy)  # mean loss, not one per example
    rec.start()
    with pytest.raises(ValueError, match="one loss per example"):
        rec.record()


def test_loss_recorder_random_subset():
    net, opt, X, y = setup()
    rec = Recorder(net, loss_on=(X, y), n_examples=20, seed=1)
    Xs, ys = rec.loss_on
    assert Xs.shape == (20, 10) and ys.shape == (20,)
    rows = [(X == x).all(dim=1).nonzero().item() for x in Xs]  # every kept example is a row of X ...
    assert len(set(rows)) == 20  # ... with no repeats
    torch.testing.assert_close(ys, y[rows])  # and keeps its own label
    torch.testing.assert_close(Recorder(net, loss_on=(X, y), n_examples=20, seed=1).loss_on[0], Xs)  # same seed
    rec.start()
    rec.record()
    with torch.no_grad():
        torch.testing.assert_close(rec.traj[0], F.cross_entropy(net(Xs), ys, reduction="none"))
    with pytest.raises(ValueError, match="n_examples"):
        Recorder(net, loss_on=(X, y), n_examples=100)  # X has only 64 examples
