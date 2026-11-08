"""Load the public backbone and preserve its classifier bias."""

import re

import torch


def load_backbone(cfg):
    spec = cfg["model"]
    if not re.fullmatch(r"[0-9a-f]{40}", spec["revision"]):
        raise ValueError("The model source must use a full commit hash")
    model = torch.hub.load(
        f'{spec["repository"]}:{spec["revision"]}', spec["entrypoint"],
        pretrained=True, trust_repo=True,
    )
    if (model.fc.in_features, model.fc.out_features) != (spec["feature_dim"], spec["classes"]):
        raise ValueError("Unexpected classifier dimensions")
    model.requires_grad_(False)
    return model.eval().to(cfg["runtime"]["device"])


def head_state(model):
    # Paper uses x @ W; PyTorch Linear stores the transpose of W.
    bias = model.fc.bias
    return {"weight": model.fc.weight.detach().cpu().T.contiguous(),
            "bias": bias.detach().cpu().clone() if bias is not None
            else torch.zeros(model.fc.out_features)}


def checkpoint_hash(model):
    import hashlib
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
