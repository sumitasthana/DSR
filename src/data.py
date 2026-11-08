"""Deterministic CIFAR-100 loading and complete-split feature caching."""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def cifar100_loader(cfg, split):
    if split not in ("train", "test"):
        raise ValueError(f"Unknown split: {split}")
    options = cfg["data"]
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(options["mean"], options["std"]),
    ])
    dataset = datasets.CIFAR100(options["root"], train=split == "train",
                               download=options["download"], transform=transform)
    return DataLoader(dataset, batch_size=options["batch_size"], shuffle=False,
                      num_workers=options["workers"])


@torch.inference_mode()
def extract_features(model, loader, cfg):
    """Capture exactly the inputs to the frozen head and check output parity."""
    model.eval()
    features, labels = [], []
    captured = []
    hook = model.fc.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    try:
        for images, targets in loader:
            captured.clear()
            logits = model(images.to(cfg["runtime"]["device"]))
            if len(captured) != 1:
                raise ValueError("Expected one classifier call per batch")
            batch = captured[0]
            torch.testing.assert_close(model.fc(batch), logits,
                                       atol=cfg["cache"]["parity_atol"],
                                       rtol=cfg["cache"]["parity_rtol"])
            features.append(batch.cpu())
            labels.append(targets.cpu())
    finally:
        hook.remove()
    if not features:
        raise ValueError("Cannot cache an empty dataset")
    result = {"features": torch.cat(features), "labels": torch.cat(labels)}
    validate_cache(result, cfg, len(loader.dataset))
    return result


def validate_cache(cache, cfg, expected_count):
    x, y = cache["features"], cache["labels"]
    if x.shape != (expected_count, cfg["model"]["feature_dim"]) or y.shape != (expected_count,):
        raise ValueError("Cache dimensions do not match the dataset")
    if not torch.isfinite(x).all() or y.dtype != torch.int64:
        raise ValueError("Invalid features or labels")
    if expected_count == 0 or y.min() < 0 or y.max() >= cfg["model"]["classes"]:
        raise ValueError("Invalid label range")
