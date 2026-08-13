"""Checkpoint load/save helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

import bdh


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    config: bdh.BDHConfig | dict[str, Any],
    meta: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = config.__dict__ if hasattr(config, "__dict__") else dict(config)
    payload = {
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "meta": meta or {},
    }
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | Path,
    device: torch.device | str | None = None,
    *,
    compile_model: bool = False,
) -> tuple[bdh.BDH, bdh.BDHConfig, dict[str, Any]]:
    path = Path(path)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = bdh.BDHConfig(**payload["config"])
    model = bdh.BDH(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    if compile_model:
        try:
            model = torch.compile(model)
        except Exception:
            pass
    return model, cfg, payload.get("meta", {})


def clone_model(model: bdh.BDH) -> bdh.BDH:
    """Independent copy with identical weights."""
    cfg = model.config
    clone = bdh.BDH(cfg).to(next(model.parameters()).device)
    clone.load_state_dict(model.state_dict())
    clone.eval()
    return clone
