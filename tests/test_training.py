import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.nn.utils import vector_to_parameters

from phdim import distance_matrix
from phdim.training import evaluate, fcn, train_run


def toy_data(n=200, d=20, seed=0):
    """Random inputs whose label is the argmax of the first 10 coordinates: easy to fit."""
    X = torch.randn(n, d, generator=torch.Generator().manual_seed(seed))
    return X, X[:, :10].argmax(1)


def test_fcn_shape_and_size():
    net = fcn()
    assert net(torch.zeros(3, 784)).shape == (3, 10)
    assert sum(p.numel() for p in net.parameters()) == 784 * 100 + 3 * 100 * 100 + 100 * 10


def test_evaluate_does_not_depend_on_chunk_size():
    X, y = toy_data()
    net = fcn(d_in=20)
    acc, loss = evaluate(net, X, y)
    acc_c, loss_c = evaluate(net, X, y, chunk=7)
    assert acc == acc_c
    np.testing.assert_allclose(loss, loss_c, rtol=1e-6)


def test_train_run_records_the_iterates():
    data = toy_data()
    r = train_run(data, data, lr=0.1, batch_size=20, n_iterates=10, max_iters=2000, eval_every=20, keep_traj=True)
    assert not r["diverged"] and r["converged"]
    assert r["train_acc"] == 100.0 and r["gap"] == 0.0
    assert r["traj"].shape == (10, sum(p.numel() for p in fcn(d_in=20).parameters()))
    np.testing.assert_allclose(r["D"], distance_matrix(r["traj"]))
    assert (r["D"][~np.eye(10, dtype=bool)] > 0).all()  # every SGD step moves the weights


def test_train_run_is_reproducible():
    data = toy_data()
    r1, r2 = (train_run(data, data, lr=0.1, batch_size=20, n_iterates=5, max_iters=100, eval_every=20)
              for _ in range(2))
    assert r1["iters"] == r2["iters"]
    np.testing.assert_array_equal(r1["D"], r2["D"])


def test_train_run_records_the_losses():
    data = toy_data()
    r = train_run(data, data, lr=0.1, batch_size=20, n_iterates=10, max_iters=2000, eval_every=20, keep_traj=True,
                  loss_on=data)
    assert r["losses"].shape == (10, 200)
    net = fcn(d_in=20)
    vector_to_parameters(r["traj"][-1], net.parameters())
    with torch.no_grad():
        torch.testing.assert_close(r["losses"][-1], F.cross_entropy(net(data[0]), data[1], reduction="none"))
    assert r["w_norm"] == pytest.approx(r["traj"][-1].norm().item())
