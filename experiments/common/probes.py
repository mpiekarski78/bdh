"""Probe and experience encoding helpers (byte-level)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def encode_bytes(text: str, device: torch.device | str | None = None) -> torch.Tensor:
    """UTF-8 bytes as LongTensor shape (1, T)."""
    data = list(text.encode("utf-8"))
    t = torch.tensor(data, dtype=torch.long)
    if device is not None:
        t = t.to(device)
    return t.unsqueeze(0)


def decode_bytes(idx: torch.Tensor) -> str:
    flat = idx.detach().to("cpu").reshape(-1).tolist()
    return bytes(flat).decode("utf-8", errors="backslashreplace")


def repeat_pattern(pattern: str, count: int) -> str:
    return pattern * count


def length_matched_neutral(length: int, fill: str = " ") -> str:
    if not fill:
        fill = " "
    # Repeat fill byte to exact length
    b = (fill.encode("utf-8") * (length + 8))[:length]
    return b.decode("latin-1")


def length_matched_noise(length: int, seed: int = 0) -> str:
    rng = torch.Generator().manual_seed(seed)
    vals = torch.randint(32, 127, (length,), generator=rng)
    return bytes(vals.tolist()).decode("latin-1")


def load_experience_file(path: str | Path) -> dict[str, Any]:
    import json

    return json.loads(Path(path).read_text())


def build_symbol_association_streams(
    *,
    symbol: str = "X",
    context: str = "A",
    target_a: str = "1",
    target_b: str = "7",
    exposures: int = 8,
    noise_seed: int = 0,
) -> dict[str, Any]:
    """Length-matched synthetic experience streams + shared probe."""
    line_a = f"{symbol} {context} {target_a}\n"
    line_b = f"{symbol} {context} {target_b}\n"
    hist_a = line_a * exposures
    hist_b = line_b * exposures
    L = len(hist_a.encode("utf-8"))
    hist_c = length_matched_neutral(L)
    hist_noise = length_matched_noise(L, seed=noise_seed)
    probe = f"{symbol} {context} "
    return {
        "history_a": hist_a,
        "history_b": hist_b,
        "history_c": hist_c,
        "history_noise": hist_noise,
        "probe": probe,
        "target_a": target_a,
        "target_b": target_b,
        "target_a_id": ord(target_a),
        "target_b_id": ord(target_b),
        "exposures": exposures,
        "history_bytes": L,
        "line_a": line_a,
        "line_b": line_b,
    }
