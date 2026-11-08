"""Reusable, verified feature-cache persistence for experiment runners."""

import json
from pathlib import Path

import torch

from .common import digest, file_hash, save_json, save_tensors


def load_or_create_cache(folder, name, identity, create, validate):
    """Reuse a verified cache or create one through caller-supplied functions.

    The caller controls extraction and validation, so persistence is independent
    of the dataset, model, and experiment. Incomplete saves are recomputed;
    corrupt completed saves fail rather than being silently trusted.
    """
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("Cache name must be a single filename component")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.pt"
    manifest_path = folder / f"{name}.json"
    if path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["sha256"] != file_hash(path) or manifest["identity"] != digest(identity):
            raise ValueError(f"Cache integrity or identity mismatch: {path}")
        cache = torch.load(path, map_location="cpu", weights_only=True)
        validate(cache)
        return cache, True
    cache = create()
    validate(cache)
    save_tensors(path, cache)
    save_json(manifest_path, {"sha256": file_hash(path), "identity": digest(identity),
                              "rows": len(cache["labels"]), "bytes": path.stat().st_size})
    return cache, False
