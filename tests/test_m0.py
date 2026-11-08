"""Offline checks for feature extraction and persisted baseline artifacts."""

import torch
import pytest
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.common import save_tensors
from src.data import extract_features
from src.evaluate import accuracy
from src.models import head_state


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.ReLU())
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(self.features(x))


def test_cached_logits_preserve_bias_and_batchnorm(tmp_path):
    torch.manual_seed(7)
    model = TinyClassifier().eval()
    x = torch.randn(9, 3)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0])
    cfg = {"runtime": {"device": "cpu"}, "model": {"feature_dim": 4, "classes": 2},
           "cache": {"parity_atol": 1e-6, "parity_rtol": 1e-6}}
    before = model.features[1].running_mean.clone()
    cache = extract_features(model, DataLoader(TensorDataset(x, y), batch_size=4), cfg)
    head = head_state(model)
    torch.testing.assert_close(cache["features"] @ head["weight"] + head["bias"], model(x))
    assert torch.equal(before, model.features[1].running_mean)
    path = tmp_path / "cache.pt"
    save_tensors(path, cache)
    restored = torch.load(path, weights_only=True)
    expected = int((model(x).argmax(1) == y).sum())
    assert accuracy(restored, head)["correct"] == expected
    assert accuracy(restored, head)["count"] == len(y)


def test_run_resumes_verified_splits_and_rejects_corruption(tmp_path, monkeypatch):
    from src import run_m0

    torch.manual_seed(11)
    model = TinyClassifier().eval()
    loader = DataLoader(TensorDataset(torch.randn(9, 3), torch.arange(9) % 2), batch_size=4)
    config = {"seeds": [0, 1, 2], "runtime": {"device": "cpu", "threads": 1},
              "data": {"root": "data"}, "model": {"feature_dim": 4, "classes": 2},
              "cache": {"root": "outputs", "parity_atol": 1e-6, "parity_rtol": 1e-6}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(run_m0, "load_backbone", lambda cfg: model)
    monkeypatch.setattr(run_m0, "cifar100_loader", lambda cfg, split: loader)
    monkeypatch.setattr(run_m0, "provenance", lambda: {"test": True})
    first = run_m0.run(path)

    def unexpected_extraction(*args):
        raise AssertionError("A verified split should not be extracted again")

    monkeypatch.setattr(run_m0, "extract_features", unexpected_extraction)
    assert run_m0.run(path)["rows"] == first["rows"]
    cached = next((tmp_path / "outputs").glob("*/train.pt"))
    with cached.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="integrity"):
        run_m0.run(path)
