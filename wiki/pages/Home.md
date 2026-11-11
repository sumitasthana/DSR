# Independent replication of OSP-LoRA

This wiki is the working record of an independent attempt to reproduce the experiments in
**"Orthogonal Subspace Projection for Continual Machine Unlearning via SVD-Based LoRA"**, by
Y. Rahulamathavan, N. Iqbal, J. Hu and S. Lambotharan, [arXiv:2604.12526](https://arxiv.org/abs/2604.12526).

The paper proposes a way to make a trained model forget one class at a time, repeatedly, without
destroying what it still knows. Each deletion gets its own small adapter, and every new adapter is
forced to read a different slice of the incoming signal from the ones before it.

We are checking whether that holds. We are not extending the method.

## Status

**In progress. Nothing here is a finished replication.** No official implementation was published,
so everything is rebuilt from the paper's text and equations. Every detail the paper omits is a
choice we made, listed on [Assumptions](Assumptions). Those choices are ours and not the authors'.

### What we have established

| Finding | Evidence |
|---|---|
| Capacity is `floor(d_in / r)` tasks. Past that the update is numerically zero. | Test T4, rank sweep over 2, 4, 8 gave budgets 32, 16, 8 |
| Lambda cannot move the free-rank curve, because scaling a matrix does not change its column space. | Test T7, curves identical across scales |
| Orthogonal input subspaces do not give non-interfering behaviour. Overlap near 1e-16 while 70% of predictions changed. | Test T5 |
| When the projection is applied decides the outcome. Constraining during training works; projecting afterwards does not. | M2, in-training 0.00% against post-hoc 63.14% on the deleted class |
| The router formula read literally is worse than doing nothing. It raises accuracy on the class it was asked to delete. | M2, 98.84% against a 95.97% baseline |
| On MNIST the paper's own method is excellent: near-total forgetting at almost no cost. | M2, 0.23% on the deleted digit while keeping 96.59% of 96.75% |

### What is still open

The central question is unresolved. The paper reports results at N = 20 and N = 30 tasks, and
reports free dimensions remaining near N = 16, neither of which is reachable under its own
equations if every update has full rank 4. Whether a *trained* update actually has full rank 4 is
the measurement that settles it, and it needs the CIFAR-100 run that has not happened yet.

Until then we can say the budget law holds and that the paper's reported numbers are inconsistent
with it for full-rank updates. We cannot say what their runs actually did. That distinction is
load-bearing and we try to keep it visible everywhere on these pages.

## Pages

- **[Paper Spec](Paper-Spec)** is the detailed model of the paper: every symbol, equation and
  claim, each tagged with how far we have verified it, plus the open discrepancy register.
- **[Replication Status](Replication-Status)** is the plain-language summary of where the project
  is and what each milestone found.
- **[Assumptions](Assumptions)** lists every detail the paper left out and what we chose instead.

Code, tests and notebooks are in the [repository](https://github.com/sumitasthana/DSR).

## A note to the authors

If you wrote this paper and something here misreads your work, please open an
[issue](https://github.com/sumitasthana/DSR/issues). We will correct the page and log the correction with its date, rather than
editing the history away. If an official implementation exists and we missed it, that would change
several of these entries and we would like to know.
