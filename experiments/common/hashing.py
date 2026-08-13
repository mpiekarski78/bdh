"""Weight hashing / checksum controls."""

from __future__ import annotations

import hashlib
from typing import Iterable

import torch


def iter_trainable_params(model: torch.nn.Module) -> Iterable[tuple[str, torch.Tensor]]:
    for name, param in model.named_parameters():
        if param.requires_grad:
            yield name, param


def hash_trainable_params(model: torch.nn.Module) -> str:
    """SHA256 over sorted trainable parameter bytes (CPU float32)."""
    h = hashlib.sha256()
    for name, param in sorted(iter_trainable_params(model), key=lambda x: x[0]):
        h.update(name.encode("utf-8"))
        tensor = param.detach().to(dtype=torch.float32, device="cpu").contiguous()
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def weights_equal(model_a: torch.nn.Module, model_b: torch.nn.Module, atol: float = 0.0) -> bool:
    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()
    if sd_a.keys() != sd_b.keys():
        return False
    for k in sd_a:
        if not torch.allclose(sd_a[k].float(), sd_b[k].float(), atol=atol, rtol=0.0):
            return False
    return True
