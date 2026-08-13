"""Prior-relative association: A / B vs empty prior and length-matched filler.

Question: does experience steer an existing completion, or only scramble the stream?
"""

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
from experiments.common.probes import (
    CLEAN_FILLER,
    build_task_streams,
    encode_bytes,
    length_matched_filler,
)
from experiments.common.run_io import init_run, write_json, write_summary
from experiments.common.seed import set_seed
from experiments.common.stateful_bdh import StatefulBDH, rho_distance


def _probe_read(model: StatefulBDH, probe: torch.Tensor) -> torch.Tensor:
    snap = model.get_state_snapshot()
    logits = model.step(probe)
    model.load_state_snapshot(snap)
    return logits


def _pair_row(logits: torch.Tensor, id_r: int, id_v: int, prefix: str) -> dict[str, float]:
    sr = association_strength(logits, id_r, foil=id_v)
    sv = association_strength(logits, id_v, foil=id_r)
    return {
        f"{prefix}_p_r": sr["p_target"],
        f"{prefix}_p_v": sv["p_target"],
        f"{prefix}_margin_r_minus_v": sr["logit_margin"],
        f"{prefix}_argmax": sr["argmax"],
        f"{prefix}_confidence": sr["confidence"],
    }


def _make(base, capture="none") -> StatefulBDH:
    m = StatefulBDH(clone_model(base), capture_level=capture)
    m.reset_dynamic_state(1)
    return m


def main() -> None:
    p = argparse.ArgumentParser(description="Prior-relative association + clean-distractor decay")
    p.add_argument("--checkpoint", type=str, default=str(REPO_ROOT / "checkpoints" / "base.pt"))
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--task", type=str, default="shakespeare_completion")
    p.add_argument("--exposures", type=str, default="1,2,4,8,16,32")
    p.add_argument("--decay-k", type=int, default=8)
    p.add_argument("--distractors", type=str, default="0,1,5,10,50,100,500")
    args = p.parse_args()

    exposure_list = [int(x) for x in args.exposures.split(",") if x.strip()]
    distractor_list = [int(x) for x in args.distractors.split(",") if x.strip()]
    set_seed(args.seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base, _, _ = load_checkpoint(args.checkpoint, device)

    run_dir = init_run(
        {
            "experiment": "prior_relative",
            "seed": args.seed,
            "checkpoint": args.checkpoint,
            "task": args.task,
            "exposures": exposure_list,
            "decay_k": args.decay_k,
            "distractors": distractor_list,
            "filler": CLEAN_FILLER,
        },
        prefix="prior",
    )

    streams0 = build_task_streams(args.task, exposures=1, noise_seed=args.seed)
    id_r, id_v = streams0["target_a_id"], streams0["target_b_id"]
    probe = encode_bytes(streams0["probe"], device)
    h0 = hash_trainable_params(base)

    # Empty prior: probe at RoPE position 0 (pretrained completion of the probe).
    prior = _make(base)
    logits_prior = _probe_read(prior, probe)
    prior_stats = _pair_row(logits_prior, id_r, id_v, "prior")
    assert hash_trainable_params(prior.model) == h0

    exposure_rows = []
    for k in exposure_list:
        streams = build_task_streams(args.task, exposures=k, noise_seed=args.seed)
        L = streams["history_bytes"]
        filler = length_matched_filler(L)

        a = _make(base)
        b = _make(base)
        m = _make(base)
        a.step(encode_bytes(streams["history_a"], device))
        b.step(encode_bytes(streams["history_b"], device))
        m.step(encode_bytes(filler, device))
        for mdl in (a, b, m):
            assert hash_trainable_params(mdl.model) == h0

        la = _probe_read(a, probe)
        lb = _probe_read(b, probe)
        lm = _probe_read(m, probe)

        row = {"exposures": k, "history_bytes": L, **prior_stats}
        row.update(_pair_row(la, id_r, id_v, "A"))
        row.update(_pair_row(lb, id_r, id_v, "B"))
        row.update(_pair_row(lm, id_r, id_v, "matched"))
        row["js_A_prior"] = output_divergence(la, logits_prior)["js"]
        row["js_B_prior"] = output_divergence(lb, logits_prior)["js"]
        row["js_A_matched"] = output_divergence(la, lm)["js"]
        row["js_B_matched"] = output_divergence(lb, lm)["js"]
        row["js_AB"] = output_divergence(la, lb, target_a=id_r, target_b=id_v)["js"]
        row["delta_p_r_A_vs_prior"] = row["A_p_r"] - row["prior_p_r"]
        row["delta_p_v_B_vs_prior"] = row["B_p_v"] - row["prior_p_v"]
        row["delta_p_r_A_vs_matched"] = row["A_p_r"] - row["matched_p_r"]
        row["delta_p_v_B_vs_matched"] = row["B_p_v"] - row["matched_p_v"]
        row["delta_margin_A_vs_prior"] = row["A_margin_r_minus_v"] - row["prior_margin_r_minus_v"]
        row["delta_margin_B_vs_prior"] = row["B_margin_r_minus_v"] - row["prior_margin_r_minus_v"]
        st = rho_distance(a.get_state_snapshot(), m.get_state_snapshot())
        row["state_l2_A_matched"] = st["l2"]
        row["state_l2_AB"] = rho_distance(a.get_state_snapshot(), b.get_state_snapshot())["l2"]
        exposure_rows.append(row)

    decay_rows = []
    streams_k = build_task_streams(args.task, exposures=args.decay_k, noise_seed=args.seed)
    for d in distractor_list:
        b = _make(base)
        b.step(encode_bytes(streams_k["history_b"], device))
        if d > 0:
            b.step(encode_bytes(length_matched_filler(d), device))
        assert hash_trainable_params(b.model) == h0
        lb = _probe_read(b, probe)
        row = {"distractors": d, "after_exposures": args.decay_k, **prior_stats}
        row.update(_pair_row(lb, id_r, id_v, "B"))
        row["js_B_prior"] = output_divergence(lb, logits_prior)["js"]
        row["delta_p_v_B_vs_prior"] = row["B_p_v"] - row["prior_p_v"]
        row["delta_margin_B_vs_prior"] = row["B_margin_r_minus_v"] - row["prior_margin_r_minus_v"]
        decay_rows.append(row)

    # Reset after k=8 of B should match empty prior (both probe at pos 0 after reset).
    b = _make(base)
    b.step(encode_bytes(streams_k["history_b"], device))
    b.reset_dynamic_state(1)
    lb_reset = _probe_read(b, probe)
    reset_js = output_divergence(lb_reset, logits_prior)["js"]

    xs = [r["exposures"] for r in exposure_rows]
    plot_multi(
        xs,
        {
            "P(r)|prior": [r["prior_p_r"] for r in exposure_rows],
            "P(r)|A": [r["A_p_r"] for r in exposure_rows],
            "P(r)|matched": [r["matched_p_r"] for r in exposure_rows],
            "P(v)|prior": [r["prior_p_v"] for r in exposure_rows],
            "P(v)|B": [r["B_p_v"] for r in exposure_rows],
        },
        path=run_dir / "plots" / "prior_probs.png",
        title="P(r)/P(v) vs prior and matched filler",
        xlabel="exposures",
        ylabel="probability",
    )
    plot_multi(
        xs,
        {
            "ΔP(v) B vs prior": [r["delta_p_v_B_vs_prior"] for r in exposure_rows],
            "ΔP(r) A vs prior": [r["delta_p_r_A_vs_prior"] for r in exposure_rows],
            "ΔP(v) B vs matched": [r["delta_p_v_B_vs_matched"] for r in exposure_rows],
            "ΔP(r) A vs matched": [r["delta_p_r_A_vs_matched"] for r in exposure_rows],
        },
        path=run_dir / "plots" / "prior_deltas.png",
        title="Association Δ vs prior / matched",
        xlabel="exposures",
        ylabel="Δ probability",
    )
    plot_multi(
        xs,
        {
            "JS(A, prior)": [r["js_A_prior"] for r in exposure_rows],
            "JS(B, prior)": [r["js_B_prior"] for r in exposure_rows],
            "JS(A, matched)": [r["js_A_matched"] for r in exposure_rows],
            "JS(B, matched)": [r["js_B_matched"] for r in exposure_rows],
        },
        path=run_dir / "plots" / "prior_js.png",
        title="Output JS vs prior and matched filler",
        xlabel="exposures",
        ylabel="JS",
    )
    xd = [r["distractors"] for r in decay_rows]
    plot_multi(
        xd,
        {
            "P(v)|B": [r["B_p_v"] for r in decay_rows],
            "P(v)|prior": [r["prior_p_v"] for r in decay_rows],
            "ΔP(v) vs prior": [r["delta_p_v_B_vs_prior"] for r in decay_rows],
        },
        path=run_dir / "plots" / "prior_decay.png",
        title=f"Forgetting after {args.decay_k}× love (clean filler)",
        xlabel="distractor bytes",
        ylabel="probability / Δ",
    )

    with (run_dir / "exposure.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(exposure_rows[0].keys()))
        w.writeheader()
        w.writerows(exposure_rows)
    with (run_dir / "decay.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(decay_rows[0].keys()))
        w.writeheader()
        w.writerows(decay_rows)

    metrics = {
        "prior": prior_stats,
        "reset_js_B_vs_prior": reset_js,
        "weights_unchanged": True,
        "exposure": exposure_rows,
        "decay": decay_rows,
    }
    write_json(run_dir / "metrics.json", metrics)

    lines = [
        "# Prior-relative association",
        "",
        f"Probe `{streams0['probe']}` → r=`{streams0['target_a']}` vs v=`{streams0['target_b']}`",
        f"Empty prior: P(r)={prior_stats['prior_p_r']:.4f}, P(v)={prior_stats['prior_p_v']:.4f}, "
        f"margin r-v={prior_stats['prior_margin_r_minus_v']:.3f}",
        f"Reset after {args.decay_k}× B vs empty prior JS={reset_js:.6g}",
        "",
        "## Exposure vs prior",
    ]
    for r in exposure_rows:
        lines.append(
            f"- k={r['exposures']}: ΔP(v) B-prior={r['delta_p_v_B_vs_prior']:.4f}, "
            f"ΔP(r) A-prior={r['delta_p_r_A_vs_prior']:.4f}, "
            f"JS(B,prior)={r['js_B_prior']:.4f}, JS(A,matched)={r['js_A_matched']:.4f}"
        )
    lines += ["", "## Clean-filler decay after B"]
    for r in decay_rows:
        lines.append(
            f"- d={r['distractors']}: P(v)={r['B_p_v']:.4f}, ΔP(v) vs prior={r['delta_p_v_B_vs_prior']:.4f}"
        )
    write_summary(run_dir, "\n".join(lines))
    print("Run dir:", run_dir)
    print("prior", prior_stats)
    print("reset_js", reset_js)


if __name__ == "__main__":
    main()
