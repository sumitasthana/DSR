"""The paper's four continual-unlearning methods, on cached features.

Dataset-agnostic: callers pass a list of forget subsets and a frozen head, so
the same code serves MNIST in M2 and CIFAR-100 in M3 and M4.

Training never depends on lambda. Lambda enters only in Eq. 1, at fusion time,
so one trained sequence can be evaluated at every lambda.
"""

import torch
from torch import nn

from .lora import LoRAHead, fuse
from .projection import SubspaceProjector

METHODS = ("static", "posthoc", "intraining", "router")
TRACKS_SUBSPACE = ("posthoc", "intraining")


def train_adapter(subset, weight, bias, projector_matrix, cfg, seed):
    """Minimise cross-entropy on the forget class with forward W + P A B.

    This is the paper's literal objective (Sec. III-B). The adapter learns to
    classify the forget class well, and Eq. 1 later subtracts it. The update is
    therefore the increment above the base model, not the base model's own
    competence, which is discrepancy D11 in paper_spec.md.
    """
    options = cfg["unlearn"]
    torch.manual_seed(seed)
    head = LoRAHead(weight, bias, projector_matrix,
                    cfg["projection"]["rank"], cfg["lora"]["init_std"])
    optimiser = torch.optim.Adam(head.parameters(), lr=options["lr"])
    features, labels = subset["features"], subset["labels"]
    count = len(labels)
    for _ in range(options["epochs"]):
        order = torch.randperm(count)
        for start in range(0, count, options["batch_size"]):
            index = order[start:start + options["batch_size"]]
            optimiser.zero_grad()
            nn.functional.cross_entropy(head(features[index]), labels[index]).backward()
            optimiser.step()
    return head


def to_projection_dtype(tensors, cfg):
    """Cast float tensors to the projector's dtype, leaving integer labels alone.

    Call this once on the head, the forget subsets and the evaluation cache
    before anything else in this module. Everything downstream then shares one
    dtype, float64 by default.

    The reason is capacity accounting. In float32 a rank-4 update carries
    singular-value noise around 1e-7 of its largest value, which clears the
    1e-8 relative rank threshold and makes the update look rank 5. The rank
    bookkeeping is the measurement here, so it runs in double precision. The
    head is one small linear layer on cached features, so the cost is nil.
    """
    dtype = getattr(torch, cfg["projection"]["dtype"])
    return {key: value.to(dtype) if value.is_floating_point() else value
            for key, value in tensors.items()}


def _check_dtype(cfg, *tensors):
    """Fail loudly on mixed precision rather than silently miscounting rank."""
    dtype = getattr(torch, cfg["projection"]["dtype"])
    wrong = [t.dtype for t in tensors if t.is_floating_point() and t.dtype is not dtype]
    if wrong:
        raise TypeError(
            f"Expected {dtype} throughout, got {wrong}. "
            "Pass inputs through to_projection_dtype() first."
        )


def run_sequence(method, forget_subsets, head, cfg, seed):
    """Train one adapter per task in order, applying the method's constraint."""
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    weight, bias = head["weight"], head["bias"]
    _check_dtype(cfg, weight, bias, *(s["features"] for s in forget_subsets))
    identity = torch.eye(weight.shape[0], dtype=weight.dtype)
    tracked = method in TRACKS_SUBSPACE
    projector = SubspaceProjector(weight.shape[0], cfg["projection"]) if tracked else None

    updates, adapters, reserved, free_rank = [], [], [], []
    for task, subset in enumerate(forget_subsets):
        available = projector.matrix.to(weight) if tracked else identity
        # Only the in-training method sees the constraint while optimising.
        trained = train_adapter(subset, weight, bias,
                                available if method == "intraining" else identity,
                                cfg, seed * 100 + task)
        factors = (trained.a.detach().clone(), trained.b.detach().clone())
        update = (available @ factors[0] @ factors[1]) if method == "posthoc" \
            else trained.effective_update().detach().clone()
        if tracked:
            reserved.append(projector.reserve(update))
            free_rank.append(projector.free_rank)
        updates.append(update)
        adapters.append(factors)
    return {"updates": updates, "adapters": adapters,
            "reserved": reserved, "free_rank": free_rank}


class Gate(nn.Module):
    """Softmax gate over N adapters plus a null slot meaning "no adapter".

    The paper writes alpha(x) = softmax(G(x)) over exactly N adapters, but also
    requires alpha_k(x) -> 0 on non-target data. A softmax over N entries always
    sums to 1, so it cannot express "none of them". The extra slot is the
    smallest change that makes the stated behaviour reachable. It is our choice,
    recorded in ASSUMPTIONS.md "M2", not something the paper specifies.
    """

    def __init__(self, d_in, tasks):
        super().__init__()
        self.linear = nn.Linear(d_in, tasks + 1)

    def weights(self, features):
        return torch.softmax(self.linear(features), dim=1)[:, 1:]


def train_router(forget_subsets, retain_features, cfg, seed):
    """Train the gate to name the task an input belongs to, or none of them."""
    options = cfg["unlearn"]
    torch.manual_seed(seed)
    _check_dtype(cfg, retain_features, *(s["features"] for s in forget_subsets))
    parts = [subset["features"] for subset in forget_subsets]
    targets = [torch.full((len(subset["labels"]),), task + 1, dtype=torch.long)
               for task, subset in enumerate(forget_subsets)]

    take = min(len(retain_features), options["router_retain_per_task"] * len(forget_subsets))
    parts.append(retain_features[torch.randperm(len(retain_features))[:take]])
    targets.append(torch.zeros(take, dtype=torch.long))

    features, labels = torch.cat(parts), torch.cat(targets)
    gate = Gate(features.shape[1], len(forget_subsets)).to(features.dtype)
    optimiser = torch.optim.Adam(gate.parameters(), lr=options["router_lr"])
    for _ in range(options["router_epochs"]):
        order = torch.randperm(len(labels))
        for start in range(0, len(labels), options["batch_size"]):
            index = order[start:start + options["batch_size"]]
            optimiser.zero_grad()
            nn.functional.cross_entropy(gate.linear(features[index]), labels[index]).backward()
            optimiser.step()
    return gate.eval().requires_grad_(False)


@torch.inference_mode()
def fused_logits(features, head, updates, strength, tasks):
    """Eq. 1 applied to the first `tasks` updates, then a forward pass."""
    return features @ fuse(head["weight"], updates[:tasks], strength) + head["bias"]


@torch.inference_mode()
def router_logits(features, head, adapters, gate, tasks):
    """h = x W_base + sum_k alpha_k(x) x A_k B_k, the paper's Sec. II formula.

    Plus sign, no lambda, exactly as written. Because the adapters were trained
    to classify the forget classes, this adds that competence back instead of
    removing it, so the router cannot unlearn. Its numbers are identical at
    every lambda, which is what the paper's own Table I shows.
    """
    logits = features @ head["weight"] + head["bias"]
    alpha = gate.weights(features)
    for task, (a, b) in enumerate(adapters[:tasks]):
        logits = logits + alpha[:, task:task + 1] * (features @ (a @ b))
    return logits
