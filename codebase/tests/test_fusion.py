import pytest
import torch

from src.lora import LoRAHead, fuse


@pytest.mark.parametrize("strength", [0.0, 0.5, 1.0, 2.0])
def test_fused_equals_explicit_subtraction(strength):
    torch.manual_seed(0)
    w = torch.randn(8, 5, dtype=torch.float64)
    bias = torch.randn(5, dtype=torch.float64)
    x = torch.randn(13, 8, dtype=torch.float64)
    updates = [torch.randn(8, 2, dtype=torch.float64) @ torch.randn(2, 5, dtype=torch.float64)
               for _ in range(3)]
    expected = x @ w + bias - strength * sum(x @ update for update in updates)
    torch.testing.assert_close(x @ fuse(w, updates, strength) + bias, expected)
    torch.testing.assert_close(fuse(w, [], strength), w)


def test_forward_uses_positive_projected_update(config):
    torch.manual_seed(0)
    p = torch.diag(torch.tensor([1., 1., 0., 0.], dtype=torch.float64))
    w = torch.randn(4, 3, dtype=torch.float64)
    bias = torch.randn(3, dtype=torch.float64)
    layer = LoRAHead(w, bias, p, 2, config["lora"]["init_std"])
    with torch.no_grad():
        layer.b.normal_()
    x = torch.randn(7, 4, dtype=torch.float64)
    torch.testing.assert_close(layer(x), x @ w + bias + x @ p @ layer.a @ layer.b)
    assert torch.count_nonzero(layer.effective_update()[2:]) == 0
