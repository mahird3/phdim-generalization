# phdim-generalization

[![tests](https://github.com/mahird3/phdim-generalization/actions/workflows/tests.yml/badge.svg)](https://github.com/mahird3/phdim-generalization/actions/workflows/tests.yml)

A re-implementation of

> T. Birdal, A. Lou, L. Guibas, U. Şimşekli. *Intrinsic Dimension, Persistent Homology and Generalization in Neural Networks.* NeurIPS 2021. [arXiv:2111.13171](https://arxiv.org/abs/2111.13171)

([official code](https://github.com/tolgabirdal/PHDimGeneralization)).

## The paper

Earlier work (Şimşekli et al., 2020) bounded the generalization gap of SGD by the intrinsic dimension of the set of weights it visits near a minimum: the smaller this dimension, the better the network generalizes. That dimension was hard to compute in practice. This paper makes it computable:

- the bound still holds with the **PH⁰ dimension**, which can be estimated from a finite set of iterates with persistent homology;
- PH⁰ lifetimes are the edge lengths of the minimum spanning tree, so the estimator is cheap: for $k$ points, $E_\alpha(k) = \sum_{e \in \mathrm{MST}} |e|^\alpha \sim k^{1-\alpha/d}$, and the slope $m$ of $\log E_\alpha$ against $\log k$ gives $d = \alpha/(1-m)$;
- across several networks and datasets, this dimension correlates with the generalization gap.

The paper also uses the dimension as a regularizer during training; that part is not implemented here.

## Usage

```bash
pip install -e ".[test]"
pytest
```

```python
import numpy as np
from phdim import ph_dim, distance_matrix

X = np.random.default_rng(0).random((1000, 2))   # uniform points in a square
ph_dim(X)                                         # about 1.9
ph_dim(D=distance_matrix(X), alpha=0.5)           # the distance matrix can be reused
```

To get the dimension of your own training run, attach a `Recorder` to the optimizer; your loop stays as it is:

```python
from phdim import Recorder, ph_dim

rec = Recorder(model, n_iterates=200).attach(optimizer)   # records after every optimizer.step()
...                                                       # your training; call rec.start() near the minimum
ph_dim(D=rec.distance_matrix())
```

`rec.record()` records by hand, and `@rec.track` decorates a step function.

Modules: `estimator.py` (distance matrix on CPU or GPU, MST lifetimes, Algorithm 1), `recorder.py` (iterates of any training loop), `levy.py` (β-stable Lévy paths), `shapes.py` (sets of known dimension) and `training.py` (fcn-5 on MNIST with plain SGD, using the recorder). The tests check the estimator on sets of known dimension and on Lévy paths, whose dimension is β.

## fcn-5 on MNIST

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mahird3/phdim-generalization/blob/main/notebooks/dimension_vs_generalization.ipynb)

[`notebooks/dimension_vs_generalization.ipynb`](notebooks/dimension_vs_generalization.ipynb): 6 learning rates × 2 batch sizes, trained to 100% train accuracy, then 200 more iterates (about 10 minutes on a T4). First run (one seed):

| | all 12 runs | b = 50 | b = 100 |
|---|---|---|---|
| Kendall τ (dimension, gap) | +0.70 (p = 0.001) | +0.60 | +0.87 |

Higher dimension goes with a larger gap, as the paper predicts. But both are just as correlated with lr/b (τ = −0.70), so the next thing I want to check is how much of the link the ratio explains on its own.
