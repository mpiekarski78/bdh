"""State persistence / snapshot restore experiment (Phase 6)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.common.checkpoint import clone_model, load_checkpoint
from experiments.common.hashing import hash_trainable_params
from experiments.common.metrics import association_strength, output_divergence
from experiments.common.probes import build_task_streams, encode_bytes
from experiments.common.run_io import init_run, write_json, write_summary
from experiments.common.seed import set_seed
from experiments.common.stateful_bdh import StatefulBDH, rho_distance, snapshots_allclose


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=str(REPO_ROOT / "checkpoints" / "base.pt"))
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--exposures", type=int, default=16)
    p.add_argument("--task", type=str, default="shakespeare_completion")
    args = p.parse_args()

    set_seed(args.seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base, _, _ = load_checkpoint(args.checkpoint, device)
    streams = build_task_streams(args.task, exposures=args.exposures)

    run_dir = init_run(
        {
            "experiment": "persistence_restore",
            "seed": args.seed,
            "checkpoint": args.checkpoint,
            "task": args.task,
            "exposures": args.exposures,
        },
        prefix="persistence",
    )

    m = StatefulBDH(clone_model(base), capture_level="summary")
    m.reset_dynamic_state(1)
    h0 = hash_trainable_params(m.model)
    m.step(encode_bytes(streams["history_a"], device))
    snap = m.get_state_snapshot()
    snap_path = run_dir / "state" / "S_A.pt"
    m.save_snapshot(snap_path)

    probe = encode_bytes(streams["probe"], device)
    ta, tb = streams["target_a_id"], streams["target_b_id"]
    logits_before = m.step(probe)
    strength_before = association_strength(logits_before, ta, foil=tb)

    # Reset and restore from disk in a fresh wrapper (simulates process restart)
    m2 = StatefulBDH(clone_model(base), capture_level="summary")
    m2.reset_dynamic_state(1)
    m2.load_snapshot_file(snap_path)
    assert snapshots_allclose(snap, m2.get_state_snapshot(), atol=0.0)
    logits_after = m2.step(probe)
    strength_after = association_strength(logits_after, ta, foil=tb)
    out = output_divergence(logits_before, logits_after, target_a=ta, target_b=tb)
    state_div = rho_distance(snap, m2.get_state_snapshot())  # after probe, positions advanced equally if compared wrong
    # Compare pre-probe restored state already asserted allclose

    metrics = {
        "weights_unchanged": hash_trainable_params(m.model) == h0 and hash_trainable_params(m2.model) == h0,
        "snapshot_path": str(snap_path),
        "strength_before": strength_before,
        "strength_after": strength_after,
        "output_js_before_vs_after": out["js"],
        "output_max_abs_logit_diff": float(
            (logits_before.float() - logits_after.float()).abs().max().item()
        ),
        "restore_exact": out["js"] < 1e-10,
    }
    write_json(run_dir / "metrics.json", metrics)
    write_summary(
        run_dir,
        f"""# Persistence / restore

- snapshot: `{snap_path}`
- weights_unchanged: {metrics['weights_unchanged']}
- JS(before, after restore): {out['js']}
- max abs logit diff: {metrics['output_max_abs_logit_diff']}
- P(target) before: {strength_before['p_target']}
- P(target) after: {strength_after['p_target']}
""",
    )
    print(metrics)


if __name__ == "__main__":
    main()
