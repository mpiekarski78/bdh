"""Main experience-driven state divergence experiment (A vs B vs C + controls)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from analysis.plots import plot_multi, plot_xy
from experiments.common.checkpoint import clone_model, load_checkpoint
from experiments.common.hashing import hash_trainable_params
from experiments.common.metrics import activation_divergence, association_strength, output_divergence
from experiments.common.probes import build_task_streams, encode_bytes
from experiments.common.run_io import init_run, write_json, write_summary
from experiments.common.seed import set_seed
from experiments.common.stateful_bdh import StatefulBDH, rho_distance


def _probe_bundle(model: StatefulBDH, probe: torch.Tensor, target_a: int, target_b: int):
    logits = model.step(probe, return_activations=True)
    acts = model.last_activations
    return logits, acts, association_strength(logits, target_a, foil=target_b)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=str(REPO_ROOT / "checkpoints" / "base.pt"))
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--exposures", type=int, default=8)
    p.add_argument("--task", type=str, default="shakespeare_completion")
    p.add_argument("--state-capture", choices=["none", "summary", "detailed"], default="detailed")
    args = p.parse_args()

    set_seed(args.seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base, cfg, _ = load_checkpoint(args.checkpoint, device)
    streams = build_task_streams(args.task, exposures=args.exposures, noise_seed=args.seed)

    run_dir = init_run(
        {
            "experiment": "divergence",
            "seed": args.seed,
            "checkpoint": args.checkpoint,
            "experience_set": args.task,
            "probe_set": streams["probe"],
            "state_capture": args.state_capture,
            "exposures": args.exposures,
            "streams": {k: streams[k] for k in ("task", "target_a", "target_b", "history_bytes", "exposures", "probe")},
        },
        prefix="divergence",
    )

    models = {
        "A": StatefulBDH(clone_model(base), capture_level=args.state_capture),
        "B": StatefulBDH(clone_model(base), capture_level=args.state_capture),
        "C": StatefulBDH(clone_model(base), capture_level=args.state_capture),
        "N": StatefulBDH(clone_model(base), capture_level=args.state_capture),  # noise
        "A2": StatefulBDH(clone_model(base), capture_level=args.state_capture),  # same-exp control
    }
    for m in models.values():
        m.reset_dynamic_state(1)

    hashes_before = {k: hash_trainable_params(m.model) for k, m in models.items()}

    # Verify initial identity
    init_div = rho_distance(models["A"].get_state_snapshot(), models["B"].get_state_snapshot())

    line_a = streams["line_a"]
    line_b = streams["line_b"]
    # Neutral/noise chunked per exposure for time series (byte-aligned).
    chunk_len = len(line_a.encode("latin-1"))
    hist_c = streams["history_c"].encode("latin-1")
    hist_n = streams["history_noise"].encode("latin-1")

    rows = []
    probe = encode_bytes(streams["probe"], device)
    ta, tb = streams["target_a_id"], streams["target_b_id"]

    # t0
    snapA = models["A"].get_state_snapshot()
    snapB = models["B"].get_state_snapshot()
    d = rho_distance(snapA, snapB)
    rows.append({"step": 0, "state_l2_AB": d["l2"], "state_cosine_AB": d["cosine"], "js_AB": 0.0})

    for step in range(1, args.exposures + 1):
        models["A"].step(encode_bytes(line_a, device))
        models["B"].step(encode_bytes(line_b, device))
        # C and N consume matched-length chunks
        start = (step - 1) * chunk_len
        models["C"].step(encode_bytes(hist_c[start : start + chunk_len].decode("latin-1"), device))
        models["N"].step(encode_bytes(hist_n[start : start + chunk_len].decode("latin-1"), device))
        models["A2"].step(encode_bytes(line_a, device))

        d_ab = rho_distance(models["A"].get_state_snapshot(), models["B"].get_state_snapshot())
        d_aa2 = rho_distance(models["A"].get_state_snapshot(), models["A2"].get_state_snapshot())
        d_ac = rho_distance(models["A"].get_state_snapshot(), models["C"].get_state_snapshot())
        d_an = rho_distance(models["A"].get_state_snapshot(), models["N"].get_state_snapshot())

        # Probe without permanently consuming? We need probe effect on state.
        # Snapshot, probe, restore so experience timeline stays clean.
        def probe_from(m: StatefulBDH):
            snap = m.get_state_snapshot()
            logits, acts, strength = _probe_bundle(m, probe, ta, tb)
            m.load_state_snapshot(snap)
            return logits, acts, strength

        la, acta, sa = probe_from(models["A"])
        lb, actb, sb = probe_from(models["B"])
        lc, actc, sc = probe_from(models["C"])
        out_ab = output_divergence(la, lb, target_a=ta, target_b=tb)
        act_div = activation_divergence(acta, actb)

        rows.append(
            {
                "step": step,
                "state_l2_AB": d_ab["l2"],
                "state_cosine_AB": d_ab["cosine"],
                "state_l2_AA2": d_aa2["l2"],
                "state_l2_AC": d_ac["l2"],
                "state_l2_AN": d_an["l2"],
                "js_AB": out_ab["js"],
                "kl_AB": out_ab["kl_ab"],
                "p_target_a_on_A": sa["p_target"],
                "p_target_a_on_B": association_strength(lb, ta)["p_target"],
                "p_target_b_on_B": association_strength(lb, tb)["p_target"],
                "p_target_a_on_C": sc["p_target"],
                "active_jaccard": act_div.get("mean_active_jaccard"),
                "argmax_A": out_ab["argmax_a"],
                "argmax_B": out_ab["argmax_b"],
            }
        )

    hashes_after = {k: hash_trainable_params(m.model) for k, m in models.items()}

    finals = {}
    snaps = {}
    acts = {}
    for name in models:
        snaps[name] = models[name].get_state_snapshot()
        models[name].save_snapshot(run_dir / "state" / f"{name}_after_experience.pt")
        logits, act, strength = _probe_bundle(models[name], probe, ta, tb)
        acts[name] = act
        finals[name] = {
            "strength": strength,
            "output_vs_A": output_divergence(
                # compare later
                logits,
                logits,
                target_a=ta,
                target_b=tb,
            ),
            "logits_argmax": int(torch.softmax(logits[0, -1].float(), dim=-1).argmax()),
            "assoc": strength,
        }
        finals[name]["_logits"] = logits

    # Fill pairwise
    pairwise = {}
    for other in ("B", "C", "N", "A2"):
        pairwise[f"A_vs_{other}"] = {
            "state": rho_distance(snaps["A"], snaps[other]),
            "output": output_divergence(finals["A"]["_logits"], finals[other]["_logits"], target_a=ta, target_b=tb),
            "activation": activation_divergence(acts["A"], acts[other]),
        }

    # Control 1: reset then probe (snaps[] were taken before probing)
    models["A"].reset_dynamic_state(1)
    models["B"].reset_dynamic_state(1)
    la_reset = models["A"].step(probe)
    lb_reset = models["B"].step(probe)
    reset_out = output_divergence(la_reset, lb_reset, target_a=ta, target_b=tb)
    reset_state = rho_distance(models["A"].get_state_snapshot(), models["B"].get_state_snapshot())

    models["A"].load_state_snapshot(snaps["A"])

    weights_changed = {k: hashes_before[k] != hashes_after[k] for k in hashes_before}

    # CSV + plots
    csv_path = run_dir / "metrics_timeseries.csv"
    if rows:
        keys = sorted({k for row in rows for k in row})
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    xs = [r["step"] for r in rows]
    plot_xy(
        xs,
        [r["state_l2_AB"] for r in rows],
        path=run_dir / "plots" / "state_divergence_AB.png",
        title="State L2 divergence A vs B",
        xlabel="experience step",
        ylabel="state L2",
    )
    plot_multi(
        xs,
        {
            "A vs B": [r["state_l2_AB"] for r in rows],
            "A vs A2 (same exp)": [r.get("state_l2_AA2", 0.0) or 0.0 for r in rows],
            "A vs C (neutral)": [r.get("state_l2_AC", 0.0) or 0.0 for r in rows],
            "A vs noise": [r.get("state_l2_AN", 0.0) or 0.0 for r in rows],
        },
        path=run_dir / "plots" / "state_divergence_controls.png",
        title="State divergence with controls",
        xlabel="experience step",
        ylabel="state L2",
    )
    plot_xy(
        xs,
        [r.get("js_AB", 0.0) or 0.0 for r in rows],
        path=run_dir / "plots" / "output_js_AB.png",
        title="Output JS divergence A vs B (probe)",
        xlabel="experience step",
        ylabel="JS divergence",
    )

    metrics = {
        "initial_state_divergence": init_div,
        "weights_before": hashes_before,
        "weights_after": hashes_after,
        "weights_changed": weights_changed,
        "any_weights_changed": any(weights_changed.values()),
        "pairwise": {
            k: {
                "state_l2": v["state"]["l2"],
                "state_cosine": v["state"]["cosine"],
                "js": v["output"]["js"],
                "kl_ab": v["output"]["kl_ab"],
                "top_k_overlap": v["output"]["top_k_overlap"],
                "active_jaccard": v["activation"].get("mean_active_jaccard"),
                "p_target_a_on_left": v["output"].get("p_target_a_on_a"),
                "p_target_a_on_right": v["output"].get("p_target_a_on_b"),
                "p_target_b_on_left": v["output"].get("p_target_b_on_a"),
                "p_target_b_on_right": v["output"].get("p_target_b_on_b"),
            }
            for k, v in pairwise.items()
        },
        "associations": {k: finals[k]["assoc"] for k in finals},
        "reset_control": {
            "output_js_after_reset": reset_out["js"],
            "state_l2_after_reset_probe": reset_state["l2"],
            "argmax_a": reset_out["argmax_a"],
            "argmax_b": reset_out["argmax_b"],
        },
        "timeseries": rows,
    }
    write_json(run_dir / "metrics.json", metrics)

    ab = metrics["pairwise"]["A_vs_B"]
    write_summary(
        run_dir,
        f"""# Divergence experiment

## Setup
- same checkpoint / architecture / seed path
- exposures: {args.exposures}
- probe: `{streams['probe']}`
- targets: A→{streams['target_a']}  B→{streams['target_b']}

## Weight control
- any_weights_changed: **{metrics['any_weights_changed']}**

## Final A vs B
- state L2: {ab['state_l2']}
- state cosine: {ab['state_cosine']}
- output JS: {ab['js']}
- P(target_a | A): {ab['p_target_a_on_left']}
- P(target_a | B): {ab['p_target_a_on_right']}
- P(target_b | B): {ab['p_target_b_on_right']}
- active Jaccard: {ab['active_jaccard']}

## Same-experience control (A vs A2)
- state L2: {metrics['pairwise']['A_vs_A2']['state_l2']}
- JS: {metrics['pairwise']['A_vs_A2']['js']}

## Reset control
- JS after reset+same probe: {reset_out['js']}

## Noise / neutral
- A vs C state L2: {metrics['pairwise']['A_vs_C']['state_l2']}
- A vs N state L2: {metrics['pairwise']['A_vs_N']['state_l2']}
""",
    )
    print("Run dir:", run_dir)
    print("A vs B JS:", ab["js"], "state L2:", ab["state_l2"])


if __name__ == "__main__":
    main()
