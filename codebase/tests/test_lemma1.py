import pytest
import torch
from torch.nn import functional as F

from src.lora import LoRAHead


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_chain_rule_and_gradient_feasibility(config, seed):
    torch.manual_seed(seed)
    q = torch.linalg.qr(torch.randn(8, 3, dtype=torch.float64)).Q
    p = torch.eye(8, dtype=torch.float64) - q @ q.T
    layer = LoRAHead(torch.randn(8, 5, dtype=torch.float64), torch.randn(5, dtype=torch.float64),
                     p, config["projection"]["rank"], config["lora"]["init_std"])
    # Zero B would make the first gradient of A zero and the test vacuous.
    with torch.no_grad():
        layer.b.normal_()
    x = torch.randn(11, 8, dtype=torch.float64)
    target = torch.arange(11) % 5
    effective_a = layer.projector @ layer.a
    effective_a.retain_grad()
    loss = F.cross_entropy(x @ (layer.weight + effective_a @ layer.b) + layer.bias, target)
    loss.backward()
    assert torch.linalg.norm(layer.a.grad) > 0
    torch.testing.assert_close(layer.a.grad, p @ effective_a.grad)
    torch.testing.assert_close((torch.eye(8) - p) @ layer.a.grad,
                               torch.zeros_like(layer.a.grad), atol=1e-12, rtol=0)
    assert layer.weight.grad is None and layer.bias.grad is None


def test_zero_b_initialization(config):
    layer = LoRAHead(torch.eye(5, dtype=torch.float64), torch.zeros(5, dtype=torch.float64),
                     torch.eye(5, dtype=torch.float64), 4, config["lora"]["init_std"])
    layer(torch.ones(2, 5, dtype=torch.float64)).sum().backward()
    assert torch.count_nonzero(layer.a.grad) == 0
    assert torch.count_nonzero(layer.b.grad) > 0
