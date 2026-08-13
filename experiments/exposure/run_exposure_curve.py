"""Exposure-strength curve: association strength vs number of exposures."""

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
from experiments.common.metrics import association_strength, output_divergence
from experiments.common.probes import build_symbol_association_streams, encode_bytes
from experiments.common.run_io import init_run, write_json, write_summary
from experiments.common.seed import set_seed
from experiments.common.stateful_bdh import StatefulBDH, rho_distance


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=str(REPO_ROOT / "checkpoints" / "base.pt"))
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--exposures", type=str, default="1,2,4,8,16,32")
    args = p.parse_args()

    exposure_list = [int(x) for x in args.exposures.split(",") if x.strip()]
    set_seed(args.seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base, _, _ = load_checkpoint(args.checkpoint, device)

    run_dir = init_run(
        {
            "experiment": "exposure_curve",
            "seed": args.seed,
            "checkpoint": args.checkpoint,
            "exposures": exposure_list,
            "state_capture": "summary",
        },
        prefix="exposure",
    )

    rows = []
    for k in exposure_list:
        streams = build_symbol_association_streams(exposures=k, noise_seed=args.seed)
        a = StatefulBDH(clone_model(base), capture_level="summary")
        b = StatefulBDH(clone_model(base), capture_level="summary")
        c = StatefulBDH(clone_model(base), capture_level="summary")
        for m, hist in (
            (a, streams["history_a"]),
            (b, streams["history_b"]),
            (c, streams["history_c"]),
        ):
            m.reset_dynamic_state(1)
            h0 = hash_trainable_params(m.model)
            m.step(encode_bytes(hist, device))
            assert hash_trainable_params(m.model) == h0

        probe = encode_bytes(streams["probe"], device)
        ta, tb = streams["target_a_id"], streams["target_b_id"]
        la = a.step(probe)
        lb = b.step(probe)
        lc = c.step(probe)
        sa = association_strength(la, ta, foil=tb)
        sb = association_strength(lb, tb, foil=ta)
        sc = association_strength(lc, ta, foil=tb)
        out = output_divergence(la, lb, target_a=ta, target_b=tb)
        st = rho_distance(a.get_state_snapshot(), b.get_state_snapshot())
        rows.append(
            {
                "exposures": k,
                "p_target_a_on_A": sa["p_target"],
                "p_target_b_on_B": sb["p_target"],
                "p_target_a_on_C": sc["p_target"],
                "logit_margin_A": sa.get("logit_margin"),
                "logit_margin_B": sb.get("logit_margin"),
                "js_AB": out["js"],
                "state_l2_AB": st["l2"],
                "state_cosine_AB": st["cosine"],
            }
        )

    with (run_dir / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    xs = [r["exposures"] for r in rows]
    plot_multi(
        xs,
        {
            "P(1|A)": [r["p_target_a_on_A"] for r in rows],
            "P(7|B)": [r["p_target_b_on_B"] for r in rows],
            "P(1|C)": [r["p_target_a_on_C"] for r in rows],
        },
        path=run_dir / "plots" / "exposure_probs.png",
        title="Association strength vs exposures",
        xlabel="exposures",
        ylabel="probability",
    )
    plot_multi(
        xs,
        {"JS(A,B)": [r["js_AB"] for r in rows], "state L2": [r["state_l2_AB"] for r in rows]},
        path=run_dir / "plots" / "exposure_divergence.png",
        title="Divergence vs exposures",
        xlabel="exposures",
        ylabel="divergence",
    )
    write_json(run_dir / "metrics.json", {"rows": rows})
    write_summary(
        run_dir,
        "# Exposure curve\n\n" + "\n".join(
            f"- k={r['exposures']}: P(1|A)={r['p_target_a_on_A']:.4f}, P(7|B)={r['p_target_b_on_B']:.4f}, JS={r['js_AB']:.4f}"
            for r in rows
        ),
    )
    print(rows)


if __name__ == "__main__":
    main()
