"""Interference: A→B then A→C overwrite dynamics."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from analysis.plots import plot_multi
from experiments.common.checkpoint import clone_model, load_checkpoint
from experiments.common.hashing import hash_trainable_params
from experiments.common.metrics import association_strength
from experiments.common.probes import TASKS, encode_bytes
from experiments.common.run_io import init_run, write_json, write_summary
from experiments.common.seed import set_seed
from experiments.common.stateful_bdh import StatefulBDH


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=str(REPO_ROOT / "checkpoints" / "base.pt"))
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--initial-exposures", type=int, default=16)
    p.add_argument("--overwrite-exposures", type=str, default="0,1,2,4,8,16,32")
    p.add_argument("--task", type=str, default="shakespeare_completion")
    args = p.parse_args()

    overwrite_list = [int(x) for x in args.overwrite_exposures.split(",") if x.strip()]
    set_seed(args.seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base, _, _ = load_checkpoint(args.checkpoint, device)

    spec = TASKS[args.task]
    line_old = spec["line_a"]
    line_new = spec["line_b"]
    probe = spec["probe"]
    t_old, t_new = ord(spec["target_a"]), ord(spec["target_b"])

    run_dir = init_run(
        {
            "experiment": "interference",
            "seed": args.seed,
            "checkpoint": args.checkpoint,
            "task": args.task,
            "initial_exposures": args.initial_exposures,
            "overwrite_exposures": overwrite_list,
            "state_capture": "summary",
        },
        prefix="interference",
    )

    rows = []
    for k in overwrite_list:
        m = StatefulBDH(clone_model(base), capture_level="summary")
        m.reset_dynamic_state(1)
        h0 = hash_trainable_params(m.model)
        m.step(encode_bytes(line_old * args.initial_exposures, device))
        if k > 0:
            m.step(encode_bytes(line_new * k, device))
        assert hash_trainable_params(m.model) == h0
        logits = m.step(encode_bytes(probe, device))
        s_old = association_strength(logits, t_old, foil=t_new)
        s_new = association_strength(logits, t_new, foil=t_old)
        rows.append(
            {
                "overwrite_exposures": k,
                "p_old": s_old["p_target"],
                "p_new": s_new["p_target"],
                "logit_margin_old_minus_new": s_old.get("logit_margin"),
                "argmax": s_old["argmax"],
            }
        )

    with (run_dir / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    xs = [r["overwrite_exposures"] for r in rows]
    plot_multi(
        xs,
        {"P(old=1)": [r["p_old"] for r in rows], "P(new=7)": [r["p_new"] for r in rows]},
        path=run_dir / "plots" / "interference.png",
        title="Interference: old vs new association",
        xlabel="contradictory exposures",
        ylabel="probability",
    )
    write_json(run_dir / "metrics.json", {"rows": rows})
    write_summary(
        run_dir,
        f"# Interference\n\nTask `{args.task}`: {args.initial_exposures}× `{line_old.strip()}` then overwrite with `{line_new.strip()}`.\n\n"
        + "\n".join(f"- k={r['overwrite_exposures']}: P(old)={r['p_old']:.4f}, P(new)={r['p_new']:.4f}" for r in rows),
    )
    print(rows)


if __name__ == "__main__":
    main()
