"""MNIST two-layer MLP, training, and hidden-feature caching for M2.

The paper describes the MNIST model as "a fundamental 2-layer MLP" and says
nothing else. Every choice here is recorded in ASSUMPTIONS.md under "M2".
LoRA goes on the final layer only, mirroring the CIFAR-100 setup, so the
cached hidden activations play the same role as the cached ResNet features.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PIXELS = 28 * 28


class TwoLayerMLP(nn.Module):
    """784 -> hidden -> classes. `fc` is the layer that carries LoRA."""

    def __init__(self, hidden, classes):
        super().__init__()
        self.first = nn.Linear(PIXELS, hidden)
        self.fc = nn.Linear(hidden, classes)

    def features(self, images):
        return torch.relu(self.first(images.flatten(1)))

    def forward(self, images):
        return self.fc(self.features(images))


def mnist_loader(cfg, split, shuffle=False):
    if split not in ("train", "test"):
        raise ValueError(f"Unknown split: {split}")
    options = cfg["mnist"]
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(options["mean"], options["std"]),
    ])
    dataset = datasets.MNIST(cfg["data"]["root"], train=split == "train",
                             download=cfg["data"]["download"], transform=transform)
    return DataLoader(dataset, batch_size=options["batch_size"], shuffle=shuffle,
                      num_workers=cfg["data"]["workers"])


def train_mlp(cfg, seed):
    """Train the base model that unlearning will later act on."""
    options = cfg["mnist"]
    torch.manual_seed(seed)
    model = TwoLayerMLP(options["hidden"], options["classes"]).to(cfg["runtime"]["device"])
    optimiser = torch.optim.Adam(model.parameters(), lr=options["lr"])
    loader = mnist_loader(cfg, "train", shuffle=True)
    model.train()
    for _ in range(options["epochs"]):
        for images, targets in loader:
            optimiser.zero_grad()
            loss = nn.functional.cross_entropy(model(images), targets)
            loss.backward()
            optimiser.step()
    return model.eval().requires_grad_(False)


@torch.inference_mode()
def hidden_features(model, loader, cfg):
    """Cache the inputs to `fc`, checking that the head reproduces the logits."""
    model.eval()
    features, labels = [], []
    for images, targets in loader:
        batch = model.features(images.to(cfg["runtime"]["device"]))
        torch.testing.assert_close(model.fc(batch), model(images),
                                   atol=cfg["cache"]["parity_atol"],
                                   rtol=cfg["cache"]["parity_rtol"])
        features.append(batch.cpu())
        labels.append(targets.cpu())
    if not features:
        raise ValueError("Cannot cache an empty dataset")
    result = {"features": torch.cat(features), "labels": torch.cat(labels)}
    expected = (len(loader.dataset), cfg["mnist"]["hidden"])
    if result["features"].shape != expected:
        raise ValueError(f"Expected features {expected}, got {tuple(result['features'].shape)}")
    if not torch.isfinite(result["features"]).all():
        raise ValueError("Non-finite features")
    return result

