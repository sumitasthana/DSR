# Replication Status

> **Independent replication, in progress.** This page is part of an unaffiliated attempt to
> reproduce [arXiv:2604.12526](https://arxiv.org/abs/2604.12526). It is not the work of the paper's authors and carries no
> endorsement from them. Readings marked as discrepancies are our interpretation of the published
> text and may be wrong. Corrections are welcome in [Issues](https://github.com/sumitasthana/DSR/issues), and will be recorded
> in the revision log rather than quietly edited.


## What this project is doing

The paper is arXiv:2604.12526, a five-page letter. It proposes a way to make a trained model
forget one class at a time, many times over, without wrecking what it still knows.

Its idea is to give each deletion request its own small patch, called an adapter, and to force
each new adapter to read a different slice of the incoming signal from every adapter before it.
The claim is that this keeps the deletions from interfering with each other.

We are rebuilding the experiments independently to see whether that holds. We are not extending
the method. Everything the paper left unsaid is a choice we made and wrote down in
[Assumptions](Assumptions); none of those choices is presented as the paper's.

Where we disagree with the paper, the disagreement is logged in [Paper Spec](Paper-Spec) with a tag and
a test that would settle it.

## Where things stand

| Milestone | What it covers | State |
|---|---|---|
| M0 | Frozen CIFAR-100 backbone, cached features | Done, 2026-11-07 |
| M1 | Unit tests for the projection and fusion maths | Done |
| M2 | MNIST, 3 deletions, four methods | Done, rerun on Colab 2026-11-11 at `bb0ff09` |
| M3 | CIFAR-100, N = 5, end to end | Done, 2026-11-11 |
| M4 | Full CIFAR-100 sweep, N in {5,10,20,30} | Next |
| M5-M6 | Saturation curve, write-up | Not started |

45 tests pass on CPU.

The project has no blocking question open. The one that was blocking, whether a trained adapter
really uses up four directions, was measured in M3 and it does.

## M0: the starting model

We could not get the paper's own trained checkpoint, so we used a public CIFAR-100 ResNet-20
from `chenyaofo/pytorch-cifar-models`. It scores 68.83% on the full test set before any
deletion. The paper's own starting point was 60.39%, so absolute numbers will not line up. We
compare shapes of curves, not single figures.

The backbone is frozen, so we ran all 60,000 images through it once and saved the 64 numbers it
produces for each. Everything after that is a small linear problem on those saved numbers, which
is why most of this project runs on a laptop CPU in seconds.

Those caches are checked with SHA-256 and reload identically. Two offline tests cover the rest:
that the cached features and the head reproduce the full model's own logits, that BatchNorm
statistics stay frozen, and that a corrupted or interrupted cache is caught rather than used.

Evaluating the same fixed cache under three seeds gives three identical numbers; that confirms
the pipeline is deterministic and says nothing about robustness.

Committed and pushed as `0fda1b5`.

## M1: checking the maths before trusting any experiment

The projection rule is the whole paper, so we tested it on synthetic matrices before training
anything. The tests cover: the projector stays symmetric and idempotent, the paper's recursive
update behaves as written, rank runs out where it should, rank-deficient and zero updates are
handled, the gradient stays inside the allowed subspace, and the fused model matches the
sum form of Eq. 1.

One result matters more than the rest. With a 64-wide input and rank-4 adapters, three seeded
full-rank sequences used up all 64 directions after exactly 16 tasks. After that, an adapter has
nowhere left to read from and its update is exactly zero.

That is arithmetic, not a reproduction of the paper's Figure 3. But the paper reports results at
N = 20 and N = 30, both past 16, and reports about 23 free dimensions remaining near N = 16.
Neither is reachable under its own equations. Those are D2 and D3 in [Paper Spec](Paper-Spec), and they
are the strongest open thread in this project.

## M2: MNIST, three deletions, four methods

Ran locally on 2026-11-09. MNIST has a 128-wide hidden layer, so the budget is 32 tasks and
three deletions never come close to it. This milestone tests the methods, not the capacity
claim.

The best result found anywhere was the paper's own method at lambda 0.5: 96.59% retained against
a 96.76% starting point, with 0.23% accuracy on the deleted digit. That is near-complete
forgetting at almost no cost, and it is a genuine point in the paper's favour.

One prediction failed. We expected the projected methods to forget less deeply than plain
subtraction. They did not: the paper's method drove the deleted digit to 0.00%, exactly as plain
subtraction did. The difference was what it cost elsewhere. The paper's method kept 77.12% of
the rest; plain subtraction kept 44.00%.

That cuts against reading the CIFAR-100 result as "this method cannot forget". Here it forgets
completely. The difference between the two settings is how much capacity is left, which points
back at M1 and the budget question.

Two further findings came out of the rerun on 11 November.

The first is the only result so far that supports the paper rather than questioning it. Post-hoc
and in-training projection use the same projector and the same rank, and differ only in when the
projection is applied. At lambda 2.0 in-training is at 0.00% on the deleted digit and post-hoc is
still at 63.14%. Projecting a finished solution throws away the directions the optimiser used;
constraining from the first step lets it find a solution inside the allowed subspace. That is what
Lemma 1 is for, and it works.

The second is that the router, built from the paper's literal formula, is worse than doing
nothing. It scores 98.84% on the digit it was asked to delete, which is higher than the 95.97%
it started at, and it damages the kept digits as well. Its formula has a plus sign and no lambda,
so it adds forget-class competence instead of removing it. The paper's Table I presents the router
as a reasonable baseline. Those numbers cannot come from this formula, which is now D16.

Three seeds also showed how noisy this gets once a method does real damage: static fusion retains
44.71% at lambda 2.0 with a standard deviation of 27.91. The paper reports single numbers with no
variance, and one of its key claims rests on a 0.70-point difference.

## M3: CIFAR-100, five deletions, the measurement that mattered

Ran locally on 2026-11-11, three seeds instead of the one the milestone asks for. This is the
first time the paper's own dataset carried a full deletion sequence.

One number decides this milestone. Every adapter, once trained, used up exactly 4 of the 64
available input directions. Five tasks, two methods, three seeds, no exceptions and no variation.
The free space fell 64, 60, 56, 52, 48, 44.

That settles the question the project has been carrying since M1. The capacity limit of 16 tasks
was arithmetic before; now it is measured on real data. The paper needs the opposite: its Figure 3
reports about 23 directions still free near task 16, which would require each task to use about
2.56. Nothing here uses less than 4, and nothing in the method could, because the projection
reserves the top 4 singular directions of an update that has exactly 4.

So the paper's Figure 3 cannot be produced by the paper's own equations. That is a statement
about the published text, not about what the authors' code did, and we still cannot see their
code.

The method comparison came out mostly as the paper claims. In-training projection held more
accuracy than plain subtraction at every lambda up to 1.5: 63.77% against 58.87% at lambda 1.0,
on a 68.93% starting point. That is the paper's central claim and it holds.

Two things did not. In-training does not forget less deeply than plain subtraction; both hit
0.00% from lambda 0.5 onward, the same failure M2 found on MNIST. And at lambda 2.0 the ordering
reverses, with in-training down to 37.12% against 41.32%. One cell at the edge of the range is
not a pattern, but the paper only ever reports lambda 0.5 and 1.0, so it is worth watching.

The two findings M2 produced both reproduced here, more sharply. Post-hoc projection still cannot
forget: 56.33% on the deleted class where in-training is at 0.00%. And the router is still worse
than doing nothing, raising accuracy on the deleted classes from 67.00% to 82.67% while dropping
the rest.

One correction to our own earlier reasoning. After M2 we said seed spread grows with the amount
of damage a method does. CIFAR-100 does comparable damage at a tenth of the spread, 41.32 +/- 2.53
against MNIST's 44.71 +/- 27.91. That generalisation was read from one dataset and has been
narrowed.

## What has been verified end to end on Colab

Notebook 01 ran on Colab at commit `748d5c58` and its artifacts reached R2 under
`dsr/nb01-projection-budget-checks/`. That run confirmed the budget result and the lambda
result, and it proved the storage path works.

It did **not** use real CIFAR-100 features. The M0 cache is not in the clone, so the notebook
fell back to random stand-ins and said so in its output. Only its interference test depends on
that, which is why its headline number differs from the local run: 73.4% of predictions changed
on random inputs against 69.6% on real features. The conclusion is the same either way.

## Notes

The repository has no enclosing implementation directory. Source, tests, configuration and
package metadata sit at the root. Feature-cache persistence is a reusable library component and
`prepare_baseline(config)` takes resolved settings so a notebook can call it. All tests passed
after that move and no experiment was rerun because of it.

See [Assumptions](Assumptions) for every choice the paper left to us, and [Paper Spec](Paper-Spec) for the open disagreements with the paper. Commands and the dated run records are kept with the project and are not published here.
