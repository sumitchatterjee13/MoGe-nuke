"""Load a MoGe model from a .safetensors file.

`MoGeModel.from_pretrained` hardcodes `model.pt` + `torch.load`, so safetensors
checkpoints need their own entry point. The model config travels in the
safetensors metadata header (written by `convert_to_safetensors.py`).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file


def read_config(path: str | Path) -> dict:
    """Return the `model_config` dict stored in the safetensors metadata header."""
    with safe_open(str(path), framework="pt") as f:
        metadata = f.metadata() or {}
    if "model_config" not in metadata:
        raise KeyError(f"{path} has no `model_config` in its metadata header")
    return json.loads(metadata["model_config"])


def load_moge(path: str | Path, device: str | torch.device = "cpu"):
    """Build a v3 MoGeModel from a safetensors checkpoint and load its weights.

    Raises if any key is missing or unexpected -- a missing key would silently
    leave part of the model (typically the sparse refiner) randomly initialised.
    """
    from moge.model.v3 import MoGeModel

    model = MoGeModel(**read_config(path))
    missing, unexpected = model.load_state_dict(load_file(str(path)), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch: missing={missing}, unexpected={unexpected}")
    return model.to(device).eval()
