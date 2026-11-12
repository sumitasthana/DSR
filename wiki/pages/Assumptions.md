# Assumptions

> **Independent replication, in progress.** This page is part of an unaffiliated attempt to
> reproduce [arXiv:2604.12526](https://arxiv.org/abs/2604.12526). It is not the work of the paper's authors and carries no
> endorsement from them. Readings marked as discrepancies are our interpretation of the published
> text and may be wrong. Corrections are welcome in [Issues](https://github.com/sumitasthana/DSR/issues), and will be recorded
> in the revision log rather than quietly edited.


## How to read this file

The paper is five pages long and leaves out most of what you need to actually run its
experiments. Every gap we hit is recorded here with what we chose and why, so that a reader can
tell our decisions apart from the paper's.

Most entries follow the same shape:

- **Missing** names what the paper does not say.
- **Choice** is what we did instead.
- **Reason** is why, and what it costs.
- The last sentence usually names the `config.yaml` setting to change if you want to try
  something else.

Nothing here is a claim about what the authors did. Where a choice is ours alone and has no
basis in the paper, the entry says so outright.

The M-numbers are milestones from the project's internal instructions section 7. M0 is the frozen starting model, M1 is
the projection maths, M2 is the MNIST experiment, M3 onwards is CIFAR-100.

A few terms used throughout: an **adapter** is the small two-matrix patch applied to the last
layer, **rank** is how thin that patch is, and the **projector** is the bookkeeping that stops a
new adapter from reusing input directions an earlier one already took.

## M0: public frozen baseline

- Missing: the paper's pretrained checkpoint and training recipe. Choice: `chenyaofo/pytorch-cifar-models` CIFAR-100 ResNet-20, source commit `786c16252c0fc58ee9adac063f8337cc4a7a497a`. Reason: a public model with the required 64-by-100 head. Change `model` in `config.yaml` and rerun with a new cache identity to replace it. This checkpoint is not claimed to reproduce the paper's starting accuracy.
- Missing: preprocessing. Choice: tensor conversion and CIFAR-100 channel mean `[0.5071, 0.4867, 0.4408]`, standard deviation `[0.2675, 0.2565, 0.2761]`, with no random augmentation. Reason: deterministic reusable features. These are an explicit replication choice, not an asserted paper setting. Change `data.mean` and `data.std` to test another recipe.
- Missing: backbone evaluation policy. Choice: frozen parameters and evaluation mode, including frozen BatchNorm statistics. Reason: cached features must represent a fixed model. Changing this requires new cache generation and a separately labeled protocol.
- Missing: classifier bias treatment. Choice: retain the public checkpoint's bias unchanged. Reason: discarding it changes the baseline. PyTorch's classifier weight is transposed to the paper's input-by-output convention.
- Missing: clean metric definition. Choice: accuracy over the complete official CIFAR-100 test split before any deletion. It is not the task-dependent retained-class accuracy from Table I.
- Seed interpretation: M0 evaluates the same fixed checkpoint and the same cache under seeds 0, 1 and 2. Nothing is trained, so all three give the same number. That zero spread shows the pipeline is deterministic; it says nothing about whether the result is robust, because there is only one model behind it. Results that do need a real spread, such as the adapter runs from M2 onwards, train independently under each seed.
- Runtime choices: CPU, four threads, batch size 256, zero data-loader workers. These bound local resource use and are configurable. No training runs in M0.
- Resume granularity: a whole split is the smallest unit that can be resumed, so an interrupted run recomputes that split from the start rather than picking up partway. Each saved split carries a SHA-256 manifest and is checked on reload. The cache is identified by the configuration, the hashes of the source files, the checkpoint hash and the environment, which means any code change makes the old cache unusable. That is deliberate: a stale cache is worse than a slow rebuild.
- Storage: local `outputs/m0` for this workstation. On Colab, set `cache.root` to persistent mounted storage before running. No automatic GPU disconnect is implemented in this command-line runner.
- Source check: inspected https://arxiv.org/abs/2604.12526 on 2026-11-07; no official implementation link was visible. This is not proof that no implementation exists elsewhere.

## Pending before adapter implementation

The paper's experimental B shape conflicts with its method notation. Use B with shape rank-by-output (4-by-100) as required by A @ B. Optimizer, learning rates, training epochs, router recipe, and forget-class order remain undecided. Adapter initialization and numerical rank handling are specified below for M1.

## M1: projection and fusion math

- Missing: numerical rank threshold. Choice: count singular values above max(rank_atol, rank_rtol * largest singular value), using the thresholds in `config.yaml`. Reason: top-r SVD vectors can contain arbitrary null directions when the update has rank below r. Only nonzero directions justify reserving capacity. This is a documented interpretation of the paper, not a claim that the authors used this threshold. Change `projection.rank_atol` and `projection.rank_rtol` for sensitivity checks.
- Missing: how to keep the projector stable over many tasks. Choice: rather than subtract from the projector once per task as Eq. 3 literally says, keep a running list of the directions already taken, re-orthonormalise that list with QR after each addition, and rebuild the projector as P = I - Q Q.T in float64. Reason: the literal recursion subtracts repeatedly, and those subtractions accumulate floating-point error until the projector is no longer quite a projector. Rebuilding from a clean basis each time avoids that drift. Tests compare this with the literal recursion for full-rank feasible updates. The code rejects materially infeasible updates instead of silently changing their meaning; post-hoc callers must project explicitly.
- Missing: behavior after capacity exhaustion. Choice: reserve no directions for a numerically zero update, and expose `free_rank == 0`. The future training runner must handle this condition explicitly. Full rank-4 updates exhaust a 64-dimensional input space after 16 tasks. The tests do not explain the paper's reported 23 remaining dimensions.
- Missing: adapter initialization. Choice: A is Gaussian with standard deviation 0.01 and B is zero. Reason: the initial effective update is zero. Change `lora.init_std` for A's scale. The initial A gradient is zero; the nontrivial gradient-feasibility test therefore deliberately uses nonzero B.
- Lemma scope: the tests establish the chain-rule gradient formula and feasibility for an orthogonal P. They do not establish that coordinatewise adaptive optimizer steps remain orthogonal, nor that orthogonal updates guarantee behavioral noninterference. Forward projection constrains the effective update regardless of the raw A coordinates.
- Static fusion subtracts updates with the configured caller-supplied strength and leaves classifier bias unchanged. The adapter training forward adds its projected update, matching the paper's stated objective.

## M2: MNIST, N = 3, four methods

The paper gives MNIST two sentences. Almost everything below is a choice, not a reading.

- Missing: the MLP. Choice: 784 -> 128 -> 10, ReLU, bias on both layers, trained 3 epochs with
  Adam at 1e-3, batch 256, no augmentation. Reason: the paper says "a fundamental 2-layer MLP"
  and nothing else. Change `mnist.hidden`, `mnist.epochs`, `mnist.lr` in `config.yaml`. We
  report the clean accuracy we get and do not claim it matches theirs.
- Missing: which layer carries LoRA. Choice: the final layer only, so `W_base` is 128 by 10.
  Reason: this mirrors the CIFAR-100 setup the paper does describe, where LoRA goes on the
  classification head alone. Consequence: `d_in = 128`, so with rank 4 the capacity budget is
  32 tasks and N = 3 never approaches it. MNIST cannot test the budget.
- Missing: the forget objective. Choice: the literal reading already used in `src/lora.py`.
  Train the adapter to minimise cross-entropy on the forget class with forward
  `W_base + P A B`, then subtract it in Eq. 1. Reason: the project's internal instructions section 5 says implement the
  literal reading first. This is the mechanism recorded as D11 in [Paper Spec](Paper-Spec), and it is
  why forget accuracy is expected to floor near the base rate rather than near zero.
- Missing: adapter training recipe. Choice: Adam, learning rate 1e-2, 20 epochs per task,
  batch 128, over that task's forget-class training images only. Reason: the forget set for one
  MNIST class is about 6,000 images, so a task converges quickly. Change the `unlearn` block.
- Missing: the router. Choice: a linear gate `G: R^128 -> R^{N+1}` with softmax, where index 0
  means "no adapter" and `alpha_k = softmax(G(x))[k]` for k in 1..N. Trained with cross-entropy
  on the union of the forget sets, labelled by task index, plus an equal sample of retain data
  labelled 0. Reason: the paper writes `alpha(x) = softmax(G(x))` over N adapters, but also says
  the gate must push `alpha_k(x) -> 0` for non-target data. A softmax over exactly N entries
  cannot express "none of them", because its outputs always sum to 1. The extra null slot is the
  smallest change that makes the paper's stated behaviour reachable. Recorded because it is our
  invention, not theirs.
- Missing: how lambda applies to the router. Choice: it does not. The paper's router formula is
  `h = x W_base + sum_k alpha_k(x) x A_k B_k`, with a plus sign and no lambda. We implement that
  literally. Consequence: the router adds forget-class competence rather than removing it, so it
  is not an unlearning method at all, and its numbers must be identical at every lambda. The
  paper's own Table I shows exactly that, with the router unchanged between lambda 0.5 and 1.0.
  If this reading is wrong, the alternative is a minus sign with lambda, which we have not run.
- Missing: the definitions of retain and forget accuracy (D6 and D7). Choice: report all four
  readings for every run rather than picking one. Forget accuracy is reported both on the
  current task's class alone and averaged over every class forgotten so far. Retain accuracy is
  reported both on the test set with forgotten classes removed and on the full test set. Reason:
  the paper's trend across N means opposite things under the two forget readings, and we cannot
  tell which it used. This is test T8 in [Paper Spec](Paper-Spec).
- Forget class order: 9, then 5, then 3, as the paper states for MNIST.
- Seeds: 0, 1, 2 control adapter initialisation, router initialisation, and batch order. The
  base MLP is trained once per seed as well, so the spread is a real one across independently
  trained models, unlike the M0 evaluation spread.
- Missing: numerical precision for adapter training. Choice: run the whole unlearning pipeline
  in float64, the dtype already configured for the projector. Reason: found while writing the
  M2 tests. In float32 a genuinely rank-4 update carries singular-value noise around 1e-7 of its
  largest value, which clears the 1e-8 relative rank threshold, so the projector counts it as
  rank 5 and refuses it. Capacity accounting is the measurement this project rests on, so it
  cannot run at a precision where the noise floor sits above the threshold. The head is one
  small linear layer on cached features, so the cost is negligible. `to_projection_dtype` casts
  at the boundary and `run_sequence` raises on mixed precision rather than miscounting silently.
  The paper almost certainly trained in float32; this is our choice, not a reading of theirs.
- Withdrawn 2026-11-09: an earlier note here said a trained post-hoc update sometimes has
  effective rank 2 instead of 4, and floated it as a candidate explanation for D2. That came
  from a 16-dimensional synthetic unit-test fixture. On real MNIST features every task reserved
  exactly 4, across both projected methods and all three seeds. The observation does not
  generalise and is not evidence for D2. Test T1 on CIFAR-100 remains the measurement that
  decides it.

## M3: CIFAR-100, N = 5, four methods

This is the paper's own dataset. More of it is specified than MNIST was, but the unlearning
recipe is still entirely ours.

- Missing: which classes are deleted, and in what order. Choice: classes 0, 1, 2, 3, 4 by index,
  in that order. Reason: the paper names no classes and gives no ordering rule. The first five by
  index is the one choice that cannot be accused of selecting easy or hard classes after the
  fact. Change `cifar100.forget_classes` in `config.yaml`. Consequence: each task has 500
  training images, an order of magnitude fewer than an MNIST class, so adapters see far fewer
  gradient steps at the same epoch count.
- Missing: the adapter training recipe, again. Choice: the same `unlearn` block M2 used, Adam at
  1e-2 for 20 epochs at batch 128. Reason: changing it between datasets would make the two
  milestones incomparable, and the paper gives no recipe for either. Consequence: 500 images at
  batch 128 is 4 steps per epoch, so 80 steps per task against roughly 960 on MNIST. We have not
  checked whether the adapters are undertrained at this budget; if a later run shows they are,
  that is a confound on every M3 number and this entry is where to start.
- Missing: nothing about the backbone, which the paper does describe. Choice: the public
  `chenyaofo/pytorch-cifar-models` ResNet-20 pinned at revision `786c1625`. Reason: the paper's
  own checkpoint was never published. Consequence: our starting point is 68.83% against their
  60.39%, so absolute numbers cannot be compared with theirs. Only orderings and shapes can.
- Missing: how many seeds M3 needs. Choice: three, where the project's internal instructions section 7 asks for one.
  Reason: D14 was raised to high after M2, the run costs minutes on cached features, and a single
  number would have reproduced the exact weakness we had just criticised the paper for. Recorded
  as a deviation from the milestone, not a silent change to it.
- Not an assumption, a measurement: the capacity budget. With `d_in = 64` and rank 4 the budget
  is 16 tasks, and M3 confirmed every trained update consumes exactly 4 directions. N = 5 spends
  5 of 16. Nothing in M3 touches saturation; that is M4.

## M4: CIFAR-100, N up to 30, plus the rank sweep

- Missing: which 30 classes, and in what order. Choice: classes 0 to 29 by index, extending M3's
  rule. Reason: same as M3. The first five are identical to M3's, which is why the N = 5 slice of
  this run reproduces M3 exactly and serves as a cross-check.
- Not an assumption, a design decision: one 30-task sequence is read at N = 5, 10, 20 and 30
  instead of running four separate experiments. Deletions are sequential and causal, so the first
  k tasks of a long run are the tasks a k-task run would have trained. `tests/test_m4.py` asserts
  this rather than assuming it, for all three non-router methods and through to the reported
  metrics. The router is excluded because its gate is trained over all N adapters at once, so it
  gets a separate gate per N.
- Missing: nothing about the rank sweep, which is ours entirely. Choice: ranks 2, 4 and 8, each
  run two tasks past its own predicted budget of 32, 16 and 8, on one seed. Reason: the sweep
  measures a deterministic property of the projection, not a trained outcome, so seeds would add
  cost and no information. Two tasks past the budget is enough to show the collapse rather than
  merely reach it. Change `m4.rank_sweep` and `m4.rank_sweep_margin`.
- Consequence worth stating plainly: rank 2 needs 33 tasks to show exhaustion, more than the 30
  the main sequence uses, so the run prepares 34 classes and the main sequence takes the first 30.
  If someone raises `m4.rank_sweep` to include rank 1, the run would need 66 classes and will
  refuse rather than silently report a budget it never reached.
