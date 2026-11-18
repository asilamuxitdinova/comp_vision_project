"""Config loading, seeding and small helpers."""
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _set_dotted(cfg: dict, dotted: str, value) -> None:
    *parents, last = dotted.split(".")
    node = cfg
    for p in parents:
        node = node.setdefault(p, {})
    node[last] = value


def load_config(path, overrides=()) -> dict:
    """Load a YAML config, following `base:` inheritance, then apply key=value overrides."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    if "base" in cfg:
        cfg = _merge(load_config(path.parent / cfg.pop("base")), cfg)
    for item in overrides:
        key, _, raw = item.partition("=")
        value = yaml.safe_load(raw)  # "true" -> bool, "40" -> int, ...
        if isinstance(value, str):
            try:
                value = float(value)  # PyYAML reads "1e-4" as a string
            except ValueError:
                pass
        _set_dotted(cfg, key, value)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_dtype(cfg: dict, device: torch.device):
    """Return the autocast dtype, or None when mixed precision is off/unsupported."""
    mode = cfg["train"].get("amp", "bf16")
    if device.type != "cuda" or mode == "none":
        return None
    if mode == "bf16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def save_json(obj, path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=float))
