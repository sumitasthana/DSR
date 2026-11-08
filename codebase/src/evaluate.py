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
