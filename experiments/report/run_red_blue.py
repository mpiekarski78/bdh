"""Phase 5 headline demo: Agent Red vs Agent Blue."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.common.checkpoint import clone_model, load_checkpoint
from experiments.common.hashing import hash_trainable_params
from experiments.common.metrics import activation_divergence, association_strength, output_divergence
from experiments.common.probes import encode_bytes, pad_to_equal_bytes
from experiments.common.run_io import init_run, write_json, write_summary
from experiments.common.seed import set_seed
from experiments.common.stateful_bdh import StatefulBDH, rho_distance


def build_history(lines: list[str], exposures_per_line: int) -> str:
    # Interleave for length-matched structured lives
    out = []
    for _ in range(exposures_per_line):
        out.extend(lines)
    return "".join(out)


def _length_matched_lives(red_lines: list[str], blue_lines: list[str]) -> tuple[list[str], list[str]]:
    if len(red_lines) != len(blue_lines):
        raise ValueError("red and blue history line lists must have the same length")
    red_out, blue_out = [], []
    for r, b in zip(red_lines, blue_lines):
        rp, bp = pad_to_equal_bytes(r, b)
        red_out.append(rp)
        blue_out.append(bp)
    return red_out, blue_out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=str(REPO_ROOT / "checkpoints" / "base.pt"))
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument(
        "--dataset",
        type=str,
        default=str(REPO_ROOT / "datasets" / "synthetic" / "agent_red_blue_shakespeare.json"),
    )
    p.add_argument("--exposures-per-line", type=int, default=None)
    args = p.parse_args()

    spec = json.loads(Path(args.dataset).read_text())
    epl = args.exposures_per_line or int(spec.get("exposures_per_line", 8))

    set_seed(args.seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base, _, _ = load_checkpoint(args.checkpoint, device)

    red_lines, blue_lines = _length_matched_lives(spec["red_history_lines"], spec["blue_history_lines"])
    red_hist = build_history(red_lines, epl)
    blue_hist = build_history(blue_lines, epl)
    assert len(red_hist.encode("utf-8")) == len(blue_hist.encode("utf-8"))

    run_dir = init_run(
        {
            "experiment": "agent_red_blue",
            "seed": args.seed,
            "checkpoint": args.checkpoint,
            "dataset": args.dataset,
            "exposures_per_line": epl,
            "history_bytes": len(red_hist.encode("utf-8")),
            "state_capture": "detailed",
        },
        prefix="redblue",
    )

    red = StatefulBDH(clone_model(base), capture_level="detailed")
    blue = StatefulBDH(clone_model(base), capture_level="detailed")
    red.reset_dynamic_state(1)
    blue.reset_dynamic_state(1)

    same_arch = True
    same_weights_init = hash_trainable_params(red.model) == hash_trainable_params(blue.model)
    h_red0 = hash_trainable_params(red.model)
    h_blue0 = hash_trainable_params(blue.model)

    red.step(encode_bytes(red_hist, device))
    blue.step(encode_bytes(blue_hist, device))

    state_div = rho_distance(red.get_state_snapshot(), blue.get_state_snapshot())
    red.save_snapshot(run_dir / "state" / "red.pt")
    blue.save_snapshot(run_dir / "state" / "blue.pt")

    probe_results = {}
    for probe in spec["probe_suite"]:
        # Snapshot-isolate each probe
        sr, sb = red.get_state_snapshot(), blue.get_state_snapshot()
        lr = red.step(encode_bytes(probe, device), return_activations=True)
        ar = red.last_activations
        red.load_state_snapshot(sr)
        lb = blue.step(encode_bytes(probe, device), return_activations=True)
        ab = blue.last_activations
        blue.load_state_snapshot(sb)

        targets = spec.get("targets", {}).get(probe, {})
        t_red = ord(targets["red"]) if "red" in targets else None
        t_blue = ord(targets["blue"]) if "blue" in targets else None
        out = output_divergence(lr, lb, target_a=t_red, target_b=t_blue)
        act = activation_divergence(ar, ab)
        entry = {
            "output": out,
            "activation": {
                "mean_active_jaccard": act.get("mean_active_jaccard"),
                "mean_summary_proxy": act.get("mean_summary_proxy"),
            },
        }
        if t_red is not None:
            entry["red_assoc"] = association_strength(lr, t_red, foil=t_blue)
            entry["blue_assoc"] = association_strength(lb, t_blue, foil=t_red)
        probe_results[probe] = entry

    # Reset ablation on primary probe
    primary = spec["probe_suite"][0]
    red.reset_dynamic_state(1)
    blue.reset_dynamic_state(1)
    lr = red.step(encode_bytes(primary, device))
    lb = blue.step(encode_bytes(primary, device))
    reset_out = output_divergence(lr, lb)

    weights_changed = (
        hash_trainable_params(red.model) != h_red0 or hash_trainable_params(blue.model) != h_blue0
    )

    headline = {
        "same_original_weights": same_weights_init,
        "same_architecture": same_arch,
        "same_probe": True,
        "different_experience": True,
        "weights_changed": weights_changed,
        "dynamic_state_divergence_l2": state_div["l2"],
        "dynamic_state_cosine": state_div["cosine"],
        "primary_probe": primary,
        "activation_divergence_jaccard": probe_results[primary]["activation"]["mean_active_jaccard"],
        "output_distribution_js": probe_results[primary]["output"]["js"],
        "reset_ablates_effect": reset_out["js"] < probe_results[primary]["output"]["js"] * 0.25
        or reset_out["js"] < 1e-6,
        "js_after_reset": reset_out["js"],
    }

    metrics = {
        "headline": headline,
        "state_divergence": state_div,
        "probe_results": probe_results,
        "reset_control": reset_out,
        "history_red": red_hist,
        "history_blue": blue_hist,
    }
    write_json(run_dir / "metrics.json", metrics)

    write_summary(
        run_dir,
        f"""# Identical weights, different lives

```
same original weights: {"YES" if headline["same_original_weights"] else "NO"}
same architecture: YES
same probe: YES
different experience: YES
weights_changed: {str(weights_changed).lower()}

dynamic-state divergence (L2): {headline["dynamic_state_divergence_l2"]}
dynamic-state cosine: {headline["dynamic_state_cosine"]}
activation divergence (Jaccard): {headline["activation_divergence_jaccard"]}
output-distribution divergence (JS): {headline["output_distribution_js"]}
reset_ablates_effect: {"YES" if headline["reset_ablates_effect"] else "NO"}
JS after reset: {headline["js_after_reset"]}
```

## Primary probe `{primary}`
- Red P(target_red): {probe_results[primary].get("red_assoc", {}).get("p_target")}
- Blue P(target_blue): {probe_results[primary].get("blue_assoc", {}).get("p_target")}
""",
    )
    print(json.dumps(headline, indent=2))
    print("Run dir:", run_dir)


if __name__ == "__main__":
    main()
