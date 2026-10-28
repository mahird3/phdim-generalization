"""fcn-5 on MNIST trained with plain SGD, recording the iterates near the minimum (Section 5.1 of the paper)."""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector
from torchvision import datasets

from .estimator import distance_matrix


def load_mnist(device="cpu", root="data"):
    """MNIST as ((X_train, y_train), (X_test, y_test)), flattened and normalized like the official code.

    The whole dataset is kept on `device` (about 190 MB), so minibatches are just index lookups.
    """
    out = []
    for train in (True, False):
        ds = datasets.MNIST(root, train=train, download=True)
        X = (ds.data.reshape(len(ds.data), -1).float() / 255 - 0.1307) / 0.3081
        out.append((X.to(device), ds.targets.to(device)))
    return tuple(out)


def fcn(depth=5, width=100, d_in=784, n_classes=10):
    """depth linear layers without bias and ReLU in between (FC_MNIST in the official code)."""
    layers = [nn.Linear(d_in, width, bias=False), nn.ReLU()]
    for _ in range(depth - 2):
        layers += [nn.Linear(width, width, bias=False), nn.ReLU()]
    layers.append(nn.Linear(width, n_classes, bias=False))
    return nn.Sequential(*layers)


@torch.no_grad()
def evaluate(net, X, y, chunk=10_000):
    """(accuracy in %, mean cross-entropy) over the whole set."""
    correct, loss = 0, 0.0
    for i in range(0, len(X), chunk):
        out = net(X[i:i + chunk])
        loss += F.cross_entropy(out, y[i:i + chunk], reduction="sum").item()
        correct += (out.argmax(1) == y[i:i + chunk]).sum().item()
    return 100 * correct / len(X), loss / len(X)


def batches(n, batch_size, gen):
    """Endless stream of minibatch indices: reshuffle every epoch, drop the last partial batch."""
    while True:
        perm = torch.randperm(n, device=gen.device, generator=gen)
        for i in range(0, n - batch_size + 1, batch_size):
            yield perm[i:i + batch_size]


def sgd_step(net, opt, X, y):
    opt.zero_grad(set_to_none=True)
    F.cross_entropy(net(X), y).backward()
    opt.step()


def train_run(train, test, lr, batch_size, n_iterates=200, max_iters=100_000, eval_every=100, seed=0,
              keep_traj=False):
    """Train fcn-5 to 100% train accuracy, then record n_iterates more SGD iterates.

    train and test are (X, y) pairs of tensors on the same device; everything runs on that device.
    Returns a dict with the accuracies, the gap and the distance matrix D of the iterates
    (only "diverged" and "iters" if the loss became infinite). With keep_traj, also the iterates.
    """
    t0 = time.time()
    (X_train, y_train), (X_test, y_test) = train, test
    device = X_train.device
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    net = fcn(d_in=X_train.shape[1]).to(device)
    opt = torch.optim.SGD(net.parameters(), lr=lr)  # plain SGD: no momentum, no weight decay
    stream = batches(len(X_train), batch_size, gen)
    run = {"lr": lr, "batch_size": batch_size}

    # phase 1: train until every training example is classified correctly (or give up at max_iters)
    it = 0
    while True:
        if it % eval_every == 0:
            train_acc, train_loss = evaluate(net, X_train, y_train)
            if not np.isfinite(train_loss):
                return {**run, "diverged": True, "iters": it}
            if train_acc == 100.0 or it >= max_iters:
                converged = train_acc == 100.0
                break
        idx = next(stream)
        sgd_step(net, opt, X_train[idx], y_train[idx])
        it += 1

    # phase 2: n_iterates more steps near the minimum, saving the flattened weights after each
    traj = torch.empty(n_iterates, sum(p.numel() for p in net.parameters()), device=device)
    for t in range(n_iterates):
        idx = next(stream)
        sgd_step(net, opt, X_train[idx], y_train[idx])
        traj[t] = parameters_to_vector(net.parameters()).detach()

    train_acc, train_loss = evaluate(net, X_train, y_train)
    test_acc, _ = evaluate(net, X_test, y_test)
    run.update(diverged=not np.isfinite(train_loss), converged=converged, iters=it,
               train_acc=train_acc, test_acc=test_acc, gap=train_acc - test_acc,
               D=distance_matrix(traj), seconds=time.time() - t0)
    if keep_traj:
        run["traj"] = traj
    return run
