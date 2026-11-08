"""Configuration, provenance, and verified atomic output helpers."""

import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys

import numpy as np
import torch
import torchvision
import yaml


def load_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not cfg["seeds"] or len(set(cfg["seeds"])) != len(cfg["seeds"]):
        raise ValueError("Seeds must be nonempty and unique")
    for group in ("data", "cache"):
        cfg[group]["root"] = str((path.parent / cfg[group]["root"]).resolve())
    return cfg


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def provenance():
    root = Path(__file__).resolve().parents[2]

    def git(*args):
        result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--porcelain"),
        "source_sha256": {str(p.relative_to(root)): file_hash(p)
                          for p in sorted((root / "codebase" / "src").glob("*.py"))},
        "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "numpy": np.__version__, "cuda_available": torch.cuda.is_available(),
    }


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    if json.loads(temporary.read_text(encoding="utf-8")) != value:
        raise ValueError("JSON verification failed")
    os.replace(temporary, path)


def save_tensors(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    restored = torch.load(temporary, map_location="cpu", weights_only=True)
    for key, tensor in value.items():
        if not torch.equal(tensor, restored[key]):
            raise ValueError(f"Tensor verification failed: {key}")
    os.replace(temporary, path)
