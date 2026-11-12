"""Build the GitHub wiki pages from the day-zero operational documents.

`day-zero/` is gitignored, so this folder is the publishable copy. Run this
after changing paper_spec.md, REPORT.md or ASSUMPTIONS.md:

    python wiki/build.py           # regenerate wiki/pages/
    python wiki/build.py --push    # regenerate, then push to the GitHub wiki

The sources are written for an agent working inside the repo. A public reader
needs different framing, so this rewrites the headers and leaves the substance
alone.
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "day-zero"
WIKI = Path(__file__).resolve().parent / "pages"
ARXIV = "https://arxiv.org/abs/2604.12526"
REPO = "https://github.com/sumitasthana/DSR"
WIKI_REMOTE = "https://github.com/sumitasthana/DSR.wiki.git"

WIKI.mkdir(exist_ok=True)

BANNER = f"""> **Independent replication, in progress.** This page is part of an unaffiliated attempt to
> reproduce [arXiv:2604.12526]({ARXIV}). It is not the work of the paper's authors and carries no
> endorsement from them. Readings marked as discrepancies are our interpretation of the published
> text and may be wrong. Corrections are welcome in [Issues]({REPO}/issues), and will be recorded
> in the revision log rather than quietly edited.

"""

# ---------------------------------------------------------------- Home

home = f"""# Independent replication of OSP-LoRA

This wiki is the working record of an independent attempt to reproduce the experiments in
**"Orthogonal Subspace Projection for Continual Machine Unlearning via SVD-Based LoRA"**, by
Y. Rahulamathavan, N. Iqbal, J. Hu and S. Lambotharan, [arXiv:2604.12526]({ARXIV}).

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

Code, tests and notebooks are in the [repository]({REPO}).

## A note to the authors

If you wrote this paper and something here misreads your work, please open an
[issue]({REPO}/issues). We will correct the page and log the correction with its date, rather than
editing the history away. If an official implementation exists and we missed it, that would change
several of these entries and we would like to know.
"""

(WIKI / "Home.md").write_text(home, encoding="utf-8")

# ---------------------------------------------------------------- Paper Spec

spec = (SRC / "paper_spec.md").read_text(encoding="utf-8")

# Strip the YAML frontmatter and rebuild it as a visible table.
spec = re.sub(r"\A---\n.*?\n---\n", "", spec, flags=re.S)

header = f"""# Paper Spec

{BANNER}| | |
|---|---|
| Paper | Orthogonal Subspace Projection for Continual Machine Unlearning via SVD-Based LoRA |
| Authors | Y. Rahulamathavan, N. Iqbal, J. Hu, S. Lambotharan |
| Identifier | [arXiv:2604.12526v1 [cs.LG]]({ARXIV}) |
| Posted | 2026-04-14 |
| Status | preprint, stated as submitted to the IEEE; no v2 as of 2026-11-08 |
| Length | 5 pages, letter format |
| Official code | none found; arXiv page inspected 2026-11-07 |
| This spec covers | Sections I to V, Eqs. 1 to 3, Lemma 1, Table I, Figs. 2 and 3 |

## How to read this page

This is a model of the paper, not a summary of it. Every symbol and claim carries a tag saying how
far we have checked it:

- `[UNVERIFIED]` taken from the text or the maths, not yet tested.
- `[INFERRED: reason]` the paper omitted it and we made a documented guess.
- `[DISCREPANCY: id]` we found a contradiction, between parts of the paper or against our tests.
  The id points into the register in Section 4.
- `[VERIFIED: ref]` confirmed by a unit test, a notebook run, or a derivation. The ref names it.

Paths like `tests/test_projection.py` and `notebooks/01_projection_budget_checks.ipynb` refer to
files in the [repository]({REPO}).

A discrepancy is a claim about the published text as we read it. Several are arithmetic and we are
confident in them. Others turn on what the authors meant by an undefined term, and those could
resolve either way once the detail is known.

---
"""

# Drop the agent-facing preamble that follows the old title.
old_preamble = """# paper_spec.md

Living model of the OSP paper. Not a summary. Every symbol and claim carries an
epistemic tag. Update this file before changing code, and after any test or
re-reading changes our understanding. Log every correction in Section 5.

Tags: `[UNVERIFIED]` `[INFERRED: reason]` `[DISCREPANCY: details]` `[VERIFIED: ref]`

---
"""
assert old_preamble in spec, "preamble not found"
spec = spec.replace(old_preamble, header, 1)

spec = spec.replace("## Section 5. Revision and Errata Log",
                    "## Section 5. Revision and Errata Log\n\n"
                    "Corrections to our own earlier understanding, newest last. Nothing is removed "
                    "from this list.", 1)


def _run_record(m):
    return (f"the project's run record for 20{m.group(3)}-{m.group(2)}-{m.group(1)}, "
            f"entry {m.group(4)}")


# The dated run records live in day-zero and are not published.
spec = re.sub(r"`?(?:day-zero/experiments/)?experiment_trajectory_(\d\d)(\d\d)(\d\d)\.md`?\s+Entry (\d+)",
              _run_record, spec)

(WIKI / "Paper-Spec.md").write_text(spec, encoding="utf-8")

# ---------------------------------------------------------------- Replication Status

report = (SRC / "REPORT.md").read_text(encoding="utf-8")
report = report.replace("# Replication status", "# Replication Status\n\n" + BANNER.rstrip() + "\n", 1)
report = report.replace(
    "See `ASSUMPTIONS.md` for every choice the paper left to us, `paper_spec.md` for open\n"
    "disagreements with the paper, `M0_USAGE.md` for commands, and `experiments/` for the dated run\n"
    "records.",
    "See [Assumptions](Assumptions) for every choice the paper left to us, and "
    "[Paper Spec](Paper-Spec) for the open disagreements with the paper. Commands and the dated "
    "run records are kept with the project and are not published here.", 1)
(WIKI / "Replication-Status.md").write_text(report, encoding="utf-8")

# ---------------------------------------------------------------- Assumptions

assumptions = (SRC / "ASSUMPTIONS.md").read_text(encoding="utf-8")
assumptions = assumptions.replace("# Replication assumptions",
                                  "# Assumptions\n\n" + BANNER.rstrip() + "\n", 1)
(WIKI / "Assumptions.md").write_text(assumptions, encoding="utf-8")

# ---------------------------------------------------------------- Sidebar

(WIKI / "_Sidebar.md").write_text(f"""### Replication of [arXiv:2604.12526]({ARXIV})

- [Home](Home)
- [Replication Status](Replication-Status)
- [Paper Spec](Paper-Spec)
- [Assumptions](Assumptions)
- [Code]({REPO})
""", encoding="utf-8")

# ---------------------------------------------------------------- Cross-page links

# The sources point at each other by filename. On the wiki they are pages, and
# AGENTS.md is not published at all.
LINKS = {
    "`ASSUMPTIONS.md`": "[Assumptions](Assumptions)",
    "`paper_spec.md`": "[Paper Spec](Paper-Spec)",
    "`REPORT.md`": "[Replication Status](Replication-Status)",
    "`day-zero/AGENTS.md`": "the project's internal instructions",
    "`AGENTS.md`": "the project's internal instructions",
}

for page in ("Paper-Spec.md", "Replication-Status.md", "Assumptions.md"):
    body = (WIKI / page).read_text(encoding="utf-8")
    for old, new in LINKS.items():
        body = body.replace(old, new)
    (WIKI / page).write_text(body, encoding="utf-8")

# ---------------------------------------------------------------- Checks

problems = []
for f in sorted(WIKI.glob("*.md")):
    body = f.read_text(encoding="utf-8")
    dashes = body.count("—")
    leak = "day-zero" in body or "AGENTS.md" in body
    if dashes:
        problems.append(f"{f.name}: {dashes} em-dash(es)")
    if leak:
        problems.append(f"{f.name}: still references an unpublished file")
    print(f"{f.name:26} {len(body):7} bytes  em-dashes={dashes}  day-zero-refs={leak}")

if problems:
    sys.exit("FAILED:\n  " + "\n  ".join(problems))

# ---------------------------------------------------------------- Push

if "--push" in sys.argv:
    tmp = Path(tempfile.mkdtemp())
    clone = tmp / "wiki"
    subprocess.run(["git", "clone", WIKI_REMOTE, str(clone)], check=True)
    for f in WIKI.glob("*.md"):
        shutil.copy2(f, clone / f.name)
    subprocess.run(["git", "add", "-A"], cwd=clone, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=clone).returncode == 0:
        print("wiki already up to date")
    else:
        subprocess.run(["git", "commit", "-m", "Update the replication record"], cwd=clone, check=True)
        subprocess.run(["git", "push"], cwd=clone, check=True)
        print("pushed")
    shutil.rmtree(tmp, ignore_errors=True)
