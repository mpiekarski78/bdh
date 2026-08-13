"""Aggregate experiment run summaries into docs/experiment_report.md."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Predeclared classification thresholds (written before interpreting results).
THRESHOLDS = {
    "min_state_l2_for_effect": 1e-3,
    "min_js_for_behavioral_effect": 1e-4,
    "decay_half_life_short_max_distractors": 50,
    "persistent_if_restore_js_max": 1e-8,
    "persistent_if_decay_p_target_min_at_100": 0.05,
}


def _load_latest(prefix: str, runs_dir: Path) -> Path | None:
    cands = sorted(runs_dir.glob(f"*_{prefix}_*"), key=lambda p: p.name)
    return cands[-1] if cands else None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def classify(metrics_bundle: dict[str, Any]) -> tuple[str, str]:
    """Return (letter, rationale)."""
    div = metrics_bundle.get("divergence") or {}
    decay = metrics_bundle.get("decay") or {}
    persist = metrics_bundle.get("persistence") or {}
    redblue = metrics_bundle.get("redblue") or {}

    pairwise = (div.get("pairwise") or {}).get("A_vs_B") or {}
    state_l2 = pairwise.get("state_l2") or (redblue.get("headline") or {}).get("dynamic_state_divergence_l2") or 0.0
    js = pairwise.get("js") or (redblue.get("headline") or {}).get("output_distribution_js") or 0.0
    reset_ok = (redblue.get("headline") or {}).get("reset_ablates_effect")
    if reset_ok is None:
        rc = div.get("reset_control") or {}
        reset_ok = (rc.get("output_js_after_reset") or 1.0) < max(js * 0.25, 1e-6)

    weights_ok = not div.get("any_weights_changed", False)

    if state_l2 < THRESHOLDS["min_state_l2_for_effect"]:
        return "A", "Dynamic state barely diverged under different experience."
    if js < THRESHOLDS["min_js_for_behavioral_effect"]:
        return "A", "State changed but probe output distributions barely diverged."

    # Decay half-life estimate
    decay_rows = (decay.get("rows") or [])
    short = True
    if decay_rows:
        p0 = decay_rows[0].get("p_target") or 0.0
        p100 = None
        for r in decay_rows:
            if r.get("distractors") == 100:
                p100 = r.get("p_target")
        if p100 is not None and p0 > 0:
            short = p100 < 0.5 * p0 or p100 < THRESHOLDS["persistent_if_decay_p_target_min_at_100"]
        # if drops a lot by 50 distractors
        for r in decay_rows:
            if r.get("distractors") == THRESHOLDS["decay_half_life_short_max_distractors"]:
                if p0 > 0 and (r.get("p_target") or 0) < 0.5 * p0:
                    short = True

    restore_js = (persist.get("output_js_before_vs_after") if persist else None)
    restore_ok = restore_js is not None and restore_js <= THRESHOLDS["persistent_if_restore_js_max"]

    if weights_ok and reset_ok and restore_ok and not short and js >= THRESHOLDS["min_js_for_behavioral_effect"]:
        return (
            "D",
            "Experience-sensitive, reset-controllable, restorable state with lasting probe effects and unchanged slow weights.",
        )
    if weights_ok and reset_ok and restore_ok and not short:
        return "C", "Associations survive longer horizons / explicit restore; still not clearly developmental."
    if weights_ok and reset_ok:
        return "B", "Experience changes subsequent processing but effects decay on short distractor horizons (working-memory-like)."
    return "A", "Limited coupling from dynamic state to future probe behavior, or controls failed."


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", type=str, default=str(REPO_ROOT / "runs"))
    p.add_argument("--out", type=str, default=str(REPO_ROOT / "docs" / "experiment_report.md"))
    args = p.parse_args()
    runs_dir = Path(args.runs_dir)

    keys = {
        "baseline": "baseline",
        "determinism": "determinism",
        "divergence": "divergence",
        "exposure": "exposure",
        "decay": "decay",
        "interference": "interference",
        "persistence": "persistence",
        "redblue": "redblue",
    }
    loaded: dict[str, Any] = {}
    summaries: dict[str, str] = {}
    envs: dict[str, Any] = {}
    for name, prefix in keys.items():
        d = _load_latest(prefix, runs_dir)
        if not d:
            continue
        loaded[name] = _read_json(d / "metrics.json")
        summaries[name] = (d / "summary.md").read_text() if (d / "summary.md").exists() else ""
        envs[name] = _read_json(d / "environment.json")
        loaded[name + "_dir"] = str(d)

    letter, rationale = classify(loaded)
    env = envs.get("baseline") or envs.get("divergence") or {}

    lines = [
        "# BDH Experience-Driven State Experiment",
        "",
        "## Environment",
        "",
        f"- device: {env.get('device')}",
        f"- torch: {env.get('torch')}",
        f"- cuda: {env.get('cuda_version')}",
        f"- machine: {env.get('machine')}",
        f"- commit: {env.get('commit')}",
        f"- fork: {env.get('fork')}",
        "",
        "## Checkpoint",
        "",
        f"- baseline run: `{loaded.get('baseline_dir')}`",
        f"- metrics: `{json.dumps(loaded.get('baseline'), default=str)[:500] if loaded.get('baseline') else 'n/a'}...`",
        "",
        "## Architecture",
        "",
        "- Public Pathway BDH-GPU baseline (`bdh.py` unmodified)",
        "- Dynamic state = per-layer linear-attention ρ via `StatefulBDH`",
        "",
        "## Seeds",
        "",
        "- training seed: 1337 (upstream)",
        "- measurement seed: 12345 (experiments)",
        "",
        "---",
        "",
        "## 1. Baseline determinism",
        "",
        summaries.get("determinism") or "_not run_",
        "",
        "## 2. State mapping",
        "",
        "See [`docs/state_map.md`](state_map.md).",
        "",
        "## 3. Experience-driven divergence",
        "",
        summaries.get("divergence") or "_not run_",
        "",
        "## 4. Activation divergence",
        "",
        "Included in divergence / red-blue metrics (Jaccard of active indices when `state_capture=detailed`).",
        "",
        "## 5. Output divergence",
        "",
        "See JS/KL in divergence and red-blue sections.",
        "",
        "## 6. Exposure-strength curve",
        "",
        summaries.get("exposure") or "_not run_",
        "",
        "## 7. Memory decay",
        "",
        summaries.get("decay") or "_not run_",
        "",
        "## 8. Interference",
        "",
        summaries.get("interference") or "_not run_",
        "",
        "## 9. State persistence",
        "",
        summaries.get("persistence") or "_not run_",
        "",
        "## 10. Conclusions",
        "",
        f"### Classification: **{letter}**",
        "",
        rationale,
        "",
        "### Headline (Agent Red / Agent Blue)",
        "",
        summaries.get("redblue") or "_not run_",
        "",
        "### Thresholds used (predeclared)",
        "",
        "```json",
        json.dumps(THRESHOLDS, indent=2),
        "```",
        "",
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"Wrote {out} classification={letter}")


if __name__ == "__main__":
    main()
