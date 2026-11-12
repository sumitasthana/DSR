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
| A trained adapter really does consume all 4 directions, on the paper's own dataset. So capacity is 16 tasks, measured and not assumed. | M3, test T1: reserved rank 4 on every task, both projected methods, all 3 seeds |
| The paper's central claim holds on CIFAR-100: constraining during training keeps more than plain subtraction. | M3, 63.77% against 58.87% at lambda 1.0, from 68.93% |
| The limit is predictive, not descriptive. Change the patch size to 2, 4 or 8 and it moves to 32, 16 or 8, exactly. | M4, test T2 |
| **Past the limit the method silently stops working.** After 30 deletions the model is identical to after 16, difference exactly 0.0. Fourteen of thirty requests changed nothing and returned normally. | M4, test T3 and D3 |

### The main finding

The method has a hard capacity limit. Each deletion permanently consumes 4 of the model's 64
input directions, so after 16 deletions there is nothing left and every further request is a
no-op. It does not degrade and it does not raise an error. It returns success and leaves the
class in place.

We measured this directly. After 30 sequential deletions the model's weights differ from the
16-deletion model by exactly zero, and the 14 classes requested after the limit were still at
49.00% accuracy, down from 71.50% only through collateral damage from the earlier deletions.

The paper's headline results are at N = 20 and N = 30, both past that limit. Two shapes it
presents as strengths are what a frozen model looks like: retain accuracy that barely changes
between N = 20 and N = 30, and forget accuracy that stays high and rises with N because the
average fills with classes that were never unlearned.

### What is still open

What this does **not** establish is what the authors' own code did. No implementation was
published. Everything here is a rebuild from the text and the equations, and several details the
paper omits would change the outcome if we guessed them wrong. Our backbone also starts at 68.83%
where theirs started at 60.39%, so no absolute number here lines up with theirs.

So the honest statement is narrow and we try to keep it visible on every page: implemented
literally, the published equations produce a method that stops working after `floor(d_in / r)`
deletions without reporting it. Whether that is what the authors ran, we cannot say.

Smaller open items: a relearn probe (test T6), and the alternative reading of the router formula
that D16 names but no run has tried.

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
