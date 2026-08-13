"""Determinism / nondeterminism floor measurement."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.common.checkpoint import clone_model, load_checkpoint
from experiments.common.hashing import hash_trainable_params
from experiments.common.metrics import output_divergence
from experiments.common.probes import build_symbol_association_streams, encode_bytes
from experiments.common.run_io import init_run, write_json, write_summary
from experiments.common.seed import set_seed
from experiments.common.stateful_bdh import StatefulBDH, rho_distance


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=str(REPO_ROOT / "checkpoints" / "base.pt"))
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--exposures", type=int, default=8)
    args = p.parse_args()

    set_seed(args.seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, _ = load_checkpoint(args.checkpoint, device)
    streams = build_symbol_association_streams(exposures=args.exposures)

    run_dir = init_run(
        {
            "experiment": "determinism",
            "seed": args.seed,
            "checkpoint": args.checkpoint,
            "exposures": args.exposures,
            "state_capture": "summary",
        },
        prefix="determinism",
    )

    a = StatefulBDH(clone_model(model), capture_level="summary")
    b = StatefulBDH(clone_model(model), capture_level="summary")
    a.reset_dynamic_state(1)
    b.reset_dynamic_state(1)

    hash0_a = hash_trainable_params(a.model)
    hash0_b = hash_trainable_params(b.model)

    hist = encode_bytes(streams["history_a"], device)
    probe = encode_bytes(streams["probe"], device)

    a.step(hist)
    b.step(hist)
    snap_div = rho_distance(a.get_state_snapshot(), b.get_state_snapshot())
    la = a.step(probe)
    lb = b.step(probe)
    out_div = output_divergence(la, lb, target_a=streams["target_a_id"], target_b=streams["target_b_id"])

    metrics = {
        "same_experience_state_l2": snap_div["l2"],
        "same_experience_state_cosine": snap_div["cosine"],
        "same_experience_output_js": out_div["js"],
        "same_experience_output_kl_ab": out_div["kl_ab"],
        "weights_hash_a": hash_trainable_params(a.model),
        "weights_hash_b": hash_trainable_params(b.model),
        "weights_unchanged_a": hash_trainable_params(a.model) == hash0_a,
        "weights_unchanged_b": hash_trainable_params(b.model) == hash0_b,
        "initial_hashes_equal": hash0_a == hash0_b,
        "state_divergence": snap_div,
        "output_divergence": out_div,
    }
    write_json(run_dir / "metrics.json", metrics)
    write_summary(
        run_dir,
        f"""# Determinism floor

Identical experience on two clones.

- state L2: {snap_div['l2']}
- state cosine: {snap_div['cosine']}
- output JS: {out_div['js']}
- weights unchanged: {metrics['weights_unchanged_a'] and metrics['weights_unchanged_b']}
""",
    )
    print(metrics)


if __name__ == "__main__":
    main()
