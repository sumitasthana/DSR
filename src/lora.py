"""Linear LoRA head and static subtraction, using the paper's matrix convention."""

import torch
from torch import nn


class LoRAHead(nn.Module):
    """Train with W_base + P A B; subtract the learned update for fusion."""

    def __init__(self, weight, bias, projector, rank, init_std):
        super().__init__()
        if projector.shape != (weight.shape[0], weight.shape[0]):
            raise ValueError("Projector dimension does not match the head")
        self.register_buffer("weight", weight.detach().clone())
        self.register_buffer("bias", bias.detach().clone())
        self.register_buffer("projector", projector.detach().to(weight).clone())
        self.a = nn.Parameter(torch.empty(weight.shape[0], rank, dtype=weight.dtype, device=weight.device))
        self.b = nn.Parameter(torch.zeros(rank, weight.shape[1], dtype=weight.dtype, device=weight.device))
        nn.init.normal_(self.a, std=init_std)

    def effective_update(self):
        return self.projector @ self.a @ self.b

    def forward(self, features):
        return features @ (self.weight + self.effective_update()) + self.bias


def fuse(weight, updates, strength):
    """Apply Eq. 1; bias remains unchanged outside this weight operation."""
    total = torch.zeros_like(weight)
    for update in updates:
        if update.shape != weight.shape:
            raise ValueError("Every update must have the base weight's shape")
        total = total + update.to(weight)
    return weight - strength * total
