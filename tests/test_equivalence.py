"""Equivalence test: StatefulBDH.step vs full-sequence BDH.forward."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import bdh
from experiments.common.seed import set_seed
from experiments.common.stateful_bdh import StatefulBDH


def test_recurrent_matches_full(atol: float = 2e-4) -> None:
    set_seed(0, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Small config for speed; same structure as default
    cfg = bdh.BDHConfig(
        n_layer=2,
        n_embd=64,
        n_head=4,
        mlp_internal_dim_multiplier=32,
        dropout=0.0,
        vocab_size=256,
    )
    model = bdh.BDH(cfg).to(device).eval()
    wrapper = StatefulBDH(model, capture_level="none")

    idx = torch.randint(0, 256, (1, 32), device=device)
    with torch.no_grad():
        full_logits, _ = model(idx)
        wrapper.reset_dynamic_state(1)
        step_logits = wrapper.step(idx)

    diff = (full_logits.float() - step_logits.float()).abs()
    max_diff = float(diff.max().item())
    mean_diff = float(diff.mean().item())
    print(f"max_diff={max_diff:.6g} mean_diff={mean_diff:.6g}")
    assert max_diff < atol, f"equivalence failed: max_diff={max_diff}"


def test_snapshot_roundtrip() -> None:
    set_seed(1, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = bdh.BDHConfig(n_layer=2, n_embd=64, n_head=4, mlp_internal_dim_multiplier=32, dropout=0.0)
    model = bdh.BDH(cfg).to(device).eval()
    w = StatefulBDH(model)
    idx = torch.randint(0, 256, (1, 8), device=device)
    w.reset_dynamic_state(1)
    w.step(idx)
    snap = w.get_state_snapshot()
    logits1 = w.step(torch.tensor([[65]], device=device))
    w.reset_dynamic_state(1)
    w.load_state_snapshot(snap)
    logits2 = w.step(torch.tensor([[65]], device=device))
    assert torch.allclose(logits1, logits2, atol=1e-5)


def test_split_history_probe_matches_concat(atol: float = 2e-4) -> None:
    """Experience then probe must match one concatenated forward (the experiment path)."""
    set_seed(2, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = bdh.BDHConfig(
        n_layer=2,
        n_embd=64,
        n_head=4,
        mlp_internal_dim_multiplier=32,
        dropout=0.0,
        vocab_size=256,
    )
    model = bdh.BDH(cfg).to(device).eval()
    hist = torch.randint(0, 256, (1, 16), device=device)
    probe = torch.randint(0, 256, (1, 5), device=device)
    concat = torch.cat([hist, probe], dim=1)
    with torch.no_grad():
        full, _ = model(concat)
    w = StatefulBDH(model)
    w.reset_dynamic_state(1)
    w.step(hist)
    split = w.step(probe)
    max_diff = float((full[:, 16:] - split).abs().max().item())
    print(f"split_vs_concat max_diff={max_diff:.6g}")
    assert max_diff < atol, f"split history/probe mismatch: {max_diff}"


if __name__ == "__main__":
    test_recurrent_matches_full()
    test_snapshot_roundtrip()
    test_split_history_probe_matches_concat()
    print("OK")
