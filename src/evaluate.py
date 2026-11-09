"""Evaluate the frozen classifier on cached features."""

import torch


@torch.inference_mode()
def accuracy(cache, head):
    logits = cache["features"] @ head["weight"] + head["bias"]
    correct = (logits.argmax(dim=1) == cache["labels"]).sum().item()
    count = len(cache["labels"])
    if not count:
        raise ValueError("Accuracy is undefined on an empty split")
    return {"correct": correct, "count": count, "accuracy_percent": 100 * correct / count}


@torch.inference_mode()
def unlearning_metrics(logits, labels, forgotten, current):
    """All four readings of retain and forget accuracy.

    The paper defines neither metric (D6 and D7), and its trend across N means
    opposite things under the two forget readings. So we report both of each
    rather than choosing one. This is test T8 in paper_spec.md.
    """
    correct = logits.argmax(dim=1) == labels
    forgotten_mask = torch.isin(labels, torch.tensor(forgotten, dtype=labels.dtype))

    def percent(mask):
        total = int(mask.sum())
        if not total:
            raise ValueError("Accuracy is undefined on an empty subset")
        return 100.0 * float(correct[mask].sum()) / total

    return {
        "forget_current": percent(labels == current),
        "forget_all": percent(forgotten_mask),
        "retain_excluding_forgotten": percent(~forgotten_mask),
        "retain_full": 100.0 * float(correct.sum()) / len(labels),
    }
