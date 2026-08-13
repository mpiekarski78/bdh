"""Memory decay: association strength vs distractor steps."""

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
from experiments.common.probes import build_task_streams, encode_bytes, length_matched_shakespeare
from experiments.common.run_io import init_run, write_json, write_summary
from experiments.common.seed import set_seed
from experiments.common.stateful_bdh import StatefulBDH


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=str(REPO_ROOT / "checkpoints" / "base.pt"))
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--exposures", type=int, default=16)
    p.add_argument("--distractors", type=str, default="0,1,5,10,50,100,500")
    p.add_argument("--task", type=str, default="shakespeare_completion")
    args = p.parse_args()

    distractor_list = [int(x) for x in args.distractors.split(",") if x.strip()]
    set_seed(args.seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base, _, _ = load_checkpoint(args.checkpoint, device)
    streams = build_task_streams(args.task, exposures=args.exposures, noise_seed=args.seed)

    run_dir = init_run(
        {
            "experiment": "decay",
            "seed": args.seed,
            "checkpoint": args.checkpoint,
            "task": args.task,
            "exposures": args.exposures,
            "distractors": distractor_list,
            "distractor_source": "shakespeare",
            "state_capture": "summary",
        },
        prefix="decay",
    )

    rows = []
    ta, tb = streams["target_a_id"], streams["target_b_id"]
    probe = encode_bytes(streams["probe"], device)
    hist = encode_bytes(streams["history_a"], device)

    for d in distractor_list:
        m = StatefulBDH(clone_model(base), capture_level="summary")
        m.reset_dynamic_state(1)
        h0 = hash_trainable_params(m.model)
        m.step(hist)
        if d > 0:
            noise = length_matched_shakespeare(d, seed=args.seed + d)
            m.step(encode_bytes(noise, device))
        assert hash_trainable_params(m.model) == h0
        logits = m.step(probe)
        s = association_strength(logits, ta, foil=tb)
        rows.append(
            {
                "distractors": d,
                "p_target": s["p_target"],
                "p_foil": s.get("p_foil"),
                "logit_margin": s.get("logit_margin"),
                "argmax": s["argmax"],
                "confidence": s["confidence"],
            }
        )

    with (run_dir / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    xs = [r["distractors"] for r in rows]
    plot_multi(
        xs,
        {
            "P(target)": [r["p_target"] for r in rows],
            "P(foil)": [r["p_foil"] for r in rows],
        },
        path=run_dir / "plots" / "decay_curve.png",
        title="Memory decay vs distractor steps",
        xlabel="distractor bytes",
        ylabel="probability",
    )
    write_json(run_dir / "metrics.json", {"rows": rows})
    write_summary(
        run_dir,
        "# Memory decay\n\nAfter establishing A→1 with "
        f"{args.exposures} exposures, insert distractors then probe.\n\n"
        + "\n".join(f"- d={r['distractors']}: P(target)={r['p_target']:.4f}" for r in rows),
    )
    print(rows)


if __name__ == "__main__":
    main()
