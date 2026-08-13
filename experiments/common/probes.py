"""Probe and experience encoding helpers (byte-level)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def encode_bytes(text: str, device: torch.device | str | None = None) -> torch.Tensor:
    """Encode as raw bytes (vocab 256). Use latin-1 so byte value == token id.

    UTF-8 would split bytes 128–255 into multi-token sequences and would not
    match upstream train.py, which reads Tiny Shakespeare as uint8.
    """
    data = list(text.encode("latin-1"))
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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def pad_to_equal_bytes(a: str, b: str, pad: str = " ") -> tuple[str, str]:
    """Pad the shorter string (before a trailing newline, if any) so UTF-8 lengths match."""
    ba, bb = a.encode("utf-8"), b.encode("utf-8")
    if len(ba) == len(bb):
        return a, b
    pad_b = pad.encode("utf-8") or b" "

    def _pad(s: str, n: int) -> str:
        raw = s.encode("utf-8")
        if raw.endswith(b"\n"):
            body, nl = raw[:-1], b"\n"
        else:
            body, nl = raw, b""
        extra = n - len(raw)
        if extra <= 0:
            return s
        body = body + pad_b * extra
        return (body + nl).decode("latin-1")

    target = max(len(ba), len(bb))
    return _pad(a, target), _pad(b, target)


def length_matched_shakespeare(length: int, seed: int = 0) -> str:
    """Unrelated in-distribution bytes, skipping windows that contain the probe targets."""
    banned = (b"lord", b"love", b"my lo", b"Lord", b"Love")
    path = _repo_root() / "input.txt"
    if path.exists() and path.stat().st_size > length + 64:
        data = path.read_bytes()
        start = (seed * 997 + 10_000) % max(1, len(data) - length)
        for _ in range(64):
            chunk = data[start : start + length]
            if len(chunk) == length and not any(tok in chunk for tok in banned):
                return chunk.decode("latin-1")
            start = (start + length + 17) % max(1, len(data) - length)
    filler = b"Enter two servants with torches.\n"
    return (filler * (length // len(filler) + 2))[:length].decode("latin-1")


TASKS: dict[str, dict[str, str]] = {
    # Original synthetic alphabet (out of distribution for Shakespeare bytes).
    "symbol_association": {
        "line_a": "X A 1\n",
        "line_b": "X A 7\n",
        "probe": "X A ",
        "target_a": "1",
        "target_b": "7",
    },
    # In-distribution completions the byte LM can actually assign mass to.
    "shakespeare_completion": {
        "line_a": "my lord\n",
        "line_b": "my love\n",
        "probe": "my lo",
        "target_a": "r",
        "target_b": "v",
    },
}


def build_task_streams(
    task: str = "shakespeare_completion",
    *,
    exposures: int = 8,
    noise_seed: int = 0,
) -> dict[str, Any]:
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; choose from {sorted(TASKS)}")
    spec = TASKS[task]
    line_a, line_b = pad_to_equal_bytes(spec["line_a"], spec["line_b"])
    hist_a = line_a * exposures
    hist_b = line_b * exposures
    L = len(hist_a.encode("utf-8"))
    target_a, target_b = spec["target_a"], spec["target_b"]
    return {
        "task": task,
        "history_a": hist_a,
        "history_b": hist_b,
        "history_c": length_matched_neutral(L),
        "history_noise": length_matched_noise(L, seed=noise_seed),
        "history_shakespeare_noise": length_matched_shakespeare(L, seed=noise_seed),
        "probe": spec["probe"],
        "target_a": target_a,
        "target_b": target_b,
        "target_a_id": ord(target_a),
        "target_b_id": ord(target_b),
        "exposures": exposures,
        "history_bytes": L,
        "line_a": line_a,
        "line_b": line_b,
    }


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
    if symbol == "X" and context == "A" and target_a == "1" and target_b == "7":
        return build_task_streams("symbol_association", exposures=exposures, noise_seed=noise_seed)
    line_a = f"{symbol} {context} {target_a}\n"
    line_b = f"{symbol} {context} {target_b}\n"
    hist_a = line_a * exposures
    hist_b = line_b * exposures
    L = len(hist_a.encode("utf-8"))
    hist_c = length_matched_neutral(L)
    hist_noise = length_matched_noise(L, seed=noise_seed)
    probe = f"{symbol} {context} "
    return {
        "task": "symbol_association_custom",
        "history_a": hist_a,
        "history_b": hist_b,
        "history_c": hist_c,
        "history_noise": hist_noise,
        "history_shakespeare_noise": length_matched_shakespeare(L, seed=noise_seed),
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
