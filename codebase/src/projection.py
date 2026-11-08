"""Track an orthonormal basis of occupied input directions."""

import torch


class SubspaceProjector:
    """Reserve only numerically nonzero directions from feasible updates.

    Rebuilding I - Q Q.T after QR prevents cumulative subtraction drift.
    With mutually orthogonal task bases this equals the paper's recursion.
    """

    def __init__(self, dimension, options, device="cpu"):
        self.options = dict(options)
        self.identity = torch.eye(dimension, dtype=getattr(torch, options["dtype"]), device=device)
        self.basis = self.identity[:, :0].clone()

    @property
    def matrix(self):
        return self.identity - self.basis @ self.basis.T

    @property
    def free_rank(self):
        return self.identity.shape[0] - self.basis.shape[1]

    @torch.no_grad()
    def reserve(self, update):
        """Reserve feasible update directions, returning the effective rank used."""
        if update.ndim != 2 or update.shape[0] != self.identity.shape[0]:
            raise ValueError("Update must have shape input-by-output")
        update = update.detach().to(self.identity)
        if not torch.isfinite(update).all():
            raise ValueError("Update must be finite")
        p = self.matrix
        projected = p @ update
        tolerance = (self.options["feasibility_atol"] +
                     self.options["feasibility_rtol"] * torch.linalg.norm(update).item())
        if torch.linalg.norm(update - projected).item() > tolerance:
            raise ValueError("Update is outside the available subspace; project it before reserving")
        u, s, _ = torch.linalg.svd(projected, full_matrices=False)
        if s.numel() == 0:
            return 0
        cutoff = max(self.options["rank_atol"], self.options["rank_rtol"] * s[0].item())
        used = int((s > cutoff).sum().item())
        if used > self.options["rank"] or used > self.free_rank:
            raise ValueError("Effective update rank exceeds adapter rank or available capacity")
        if not used:
            return 0
        # Reprojection removes tiny overlap with earlier tasks before joint QR.
        new = p @ u[:, :used]
        self.basis = torch.linalg.qr(torch.cat([self.basis, new], dim=1), mode="reduced").Q
        return used
