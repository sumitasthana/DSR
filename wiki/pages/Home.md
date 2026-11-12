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

### What is still open

The capacity question is settled, and it settled against the paper. Figure 3 reports about 23
directions still free near task 16, which would need each task to consume about 2.56. Every
trained update we have measured consumes exactly 4, so Figure 3 cannot be produced by the paper's
own Eq. 3.

What remains is the difference between inferring and observing. Table I reports results at N = 20
and N = 30, both past a budget we have now measured at 16, which means those adapters should be
training in an empty subspace and producing nothing. We have not watched that happen, because no
run of ours has yet passed task 16. That is the next milestone.

So: we can say the budget is 16 and that the paper's published numbers are not reachable from its
published equations. We cannot say what the authors' own code did, because it was never released.
That distinction is load-bearing and we try to keep it visible everywhere on these pages.

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
