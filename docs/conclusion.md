# Conclusion: BDH experience-driven state

**Date:** 14 August 2026 (experiments 13 August 2026)  
**Classification:** **B — short-term adaptive memory**  
**Code:** [mpiekarski78/bdh](https://github.com/mpiekarski78/bdh) (fork of [pathwaycom/bdh](https://github.com/pathwaycom/bdh))  
**Upstream `bdh.py`:** unmodified

This is the deliverable for the first research question. It is evidence, not a product design.

---

## Verdict

Two identical public BDH models, started from the same trained weights, **do** develop different internal state and different next-byte behavior after different experience histories, **without changing slow weights**. Resetting the inference-time state (ρ) **removes** the difference.

That state is **not** a durable associative memory. It behaves like a **short, fragile residue of the recent prefix**: it can briefly steer a completion against the pretrained prior, then collapse if more tokens arrive, if the association is repeated too often, or if the probe is mixed into a longer “life.”

**Do not** treat public BDH ρ as a long-term knowledge layer. **Do** treat it as session-scale coloring of the next tokens. Anything that must survive, be inspected, or be believed should live in an **explicit store**, not in this state.

---

## Question

> Can two identical BDH models (same architecture, same trained weights, same initial recurrent/synaptic state) diverge internally and behaviorally after different experience histories, while slow weights stay fixed?

Predeclared categories (written before looking at results):

| ID | Meaning |
|----|---------|
| A | State changes, little effect on later processing |
| B | Experience changes later processing, but the effect is short-horizon / reset-cleared |
| C | Useful associations survive long gaps or explicit restore as lasting memory |
| D | Stable, controllable, persistent substrate that would justify using BDH as implicit long-term memory |

**Result: B.** Restore of ρ is technically exact (so C’s “can serialize state” piece is true), but the *association* does not reliably survive even tiny extra context, so this is not C or D.

---

## Setup (what the numbers mean)

| Item | Value |
|------|-------|
| Hardware | NVIDIA GB10, `aarch64`, CUDA 13.0 |
| Software | Python 3.12.3, PyTorch 2.13.0+cu130, bf16 train / fp32 measure |
| Checkpoint | Tiny Shakespeare byte LM, batch 32, block 512, 3000 steps, seed 1337 |
| Train loss | 5.65 → 0.11 (next-byte cross-entropy; untrained ≈ ln 256 ≈ 5.55) |
| Architecture | Public BDH-GPU: 6 layers, `n_embd=256`, 4 heads, `N=8192` |
| Dynamic state | Per-layer linear-attention accumulator ρ, plus RoPE position |
| Measurement seed | 12345, deterministic kernels, eval (dropout off) |

Experience is a **prefix** that updates ρ. The probe is the **same** byte string afterward. Controls: weight SHA256, ρ reset, identical-experience twins, length-matched filler/noise.

Recurrent ρ was checked against full-sequence `BDH.forward` (tiny model ~2e-7; this checkpoint ~2e-5). History-then-probe matches one concatenated forward.

---

## What is true

At time T0, copies A and B are the same.

After different prefixes, at T1:

```text
weights A == weights B          (SHA256 unchanged)
state  A !=  state B            (ρ L2 ≫ 0)
same probe
  → different activations
  → different output distributions
reset ρ → difference gone
save ρ / load ρ → logits identical
```

Headline measurements on the trained checkpoint:

| Experiment | Finding |
|------------|---------|
| Dedicated A vs B (`my lord` vs `my love`, 8×, probe `my lo`) | Output JS ≈ **0.57**; P(`v`\|B) ≈ **0.61**; weights unchanged; reset JS = 0 |
| Same-experience twins | State L2 = 0, JS = 0 |
| Empty prior on `my lo` | P(`r`)=**0.64**, P(`v`)=**0.18** (model already prefers “lord”) |
| `my love` vs that prior | Can raise P(`v`) (Δ up to **+0.74** at 16×) |
| `my lord` vs that prior | Does **not** raise P(`r`) (Δ ≈ **−0.5**); matched filler does the same, so this is mostly “any prefix,” not a lord memory |
| 32× repetition | Love effect **collapses** |
| Clean filler after 8× love | **1 extra byte** can drop P(`v`) from ~0.61 to ~0; the curve is not monotonic (it can rebound later) |
| Mixed Red/Blue lives | ρ still diverges a lot; the *specific* `my lo` JS drops to ~0.006 (dilution) |
| Out-of-distribution `X A 1` vs `X A 7` | Distributions still diverge (JS ≈ 0.26) but P(target) stays ~1e-5 — unreadable as association |

So experience **can** alter future cognition without retraining. It does so as **working-memory residue**, not as a fact store.

---

## What is not true

- Slow weights are not a second learning loop at inference. They did not move.
- ρ is not a reliable key–value memory for “X A 1 vs X A 7”-style facts on this checkpoint.
- More repetition is not more memory. Past a point it **destabilizes**.
- Unrelated tokens are not a gentle clock. They **scramble** the effect.
- Serializing ρ preserves the *tensor*. It does not make the *association* long-lived under new input.
- The public repo is not Pathway’s internal Sudoku model. These results apply to **this** BDH-GPU baseline.

---

## Classification rationale

| Criterion | Evidence | Pass? |
|-----------|----------|-------|
| State diverges with experience | ρ L2 ~ 10⁴–10⁵ | yes |
| Later identical probes differ | JS 0.26–0.57 on dedicated tasks | yes |
| Weights stay fixed | SHA256 | yes |
| Reset ablates the effect | JS → 0 | yes |
| State can be saved/restored | logit diff 0 | yes (mechanism) |
| Association survives long clean distractors | 1 byte can erase it | **no** |
| Controllable write of both directions vs prior | love can steer; lord cannot strengthen | **no** |
| Stable under repetition | collapse at 32× | **no** |

That is **B**, not C or D.

---

## Implication

If a larger system needs two kinds of memory:

| Role | Put it here |
|------|-------------|
| Beliefs, evidence, goals, inspectable history | Explicit store (database / log) |
| “I just saw this pattern; the next byte may shift” | BDH ρ, **session only**, discard on reset |

BDH should not be asked to hold beliefs that were never written as data. It can, at most, **tint** the next completion after the recent prefix, unreliably.

Category D (a developmental substrate that replaces or rivals explicit memory) is **not** supported.

---

## How to reproduce

```bash
# Train once (overwrites checkpoints/base.pt). Skip if the checkpoint already exists.
python -m experiments.baseline.run_baseline --checkpoint checkpoints/base.pt --no-compile

python tests/test_equivalence.py
python -m experiments.divergence.run_divergence --checkpoint checkpoints/base.pt --task shakespeare_completion
python -m experiments.prior.run_prior --checkpoint checkpoints/base.pt
python -m experiments.report.run_red_blue --checkpoint checkpoints/base.pt
python -m experiments.report.run_report
```

Numbers live in `runs/` (gitignored) and are summarized in [`experiment_report.md`](experiment_report.md), [`hardware_and_metrics.md`](hardware_and_metrics.md), and [`CHANGELOG.md`](CHANGELOG.md).

This line of work is **closed** unless a new question is posed (for example: a coin-flip prefix with no strong pretrained bias). That would be a new experiment, not a retuning of this one until a desired effect appears.
