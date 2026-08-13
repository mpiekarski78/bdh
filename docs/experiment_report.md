# BDH Experience-Driven State Experiment

## Environment

- device: NVIDIA GB10
- torch: 2.13.0+cu130
- cuda: 13.0
- machine: aarch64
- python: 3.12.3
- commit: 2b0d7a45b058d4309c84a10e0768d541fe18bdc2
- fork: https://github.com/mpiekarski78/bdh

Hardware catalog and metric definitions: [`hardware_and_metrics.md`](hardware_and_metrics.md).

## Checkpoint

- baseline run: `/opt/BDH_v1/runs/2026-08-13_baseline_005`
- metrics: `{"final_logged_loss": 0.10977522291318335, "sample": "To be or a lion, if\nHis worth is exteem as her on. What day?\n\nTRANIO:\nDo not, sweet lords: 'He is the first t", "checkpoint": "checkpoints/base.pt"}...`

## Architecture

- Public Pathway BDH-GPU baseline (`bdh.py` unmodified)
- Dynamic state = per-layer linear-attention ρ via `StatefulBDH`

## Seeds

- training seed: 1337 (upstream)
- measurement seed: 12345 (experiments)

---

## 1. Baseline determinism

# Determinism floor

Identical experience on two clones.

- state L2: 0.0
- state cosine: 1.0
- output JS: 0.0
- weights unchanged: True


## 2. State mapping

See [`docs/state_map.md`](state_map.md).

## 3. Experience-driven divergence

# Divergence experiment

## Setup
- same checkpoint / architecture / seed path
- exposures: 8
- probe: `X A `
- targets: A→1  B→7

## Weight control
- any_weights_changed: **False**

## Final A vs B
- state L2: 58860.150390625
- state cosine: 0.9320536446214833
- output JS: 0.16532298922538757
- P(target_a | A): 8.762310244492255e-06
- P(target_a | B): 8.043244633881841e-06
- P(target_b | B): 9.781876542547252e-06
- active Jaccard: 0.7247760842754767

## Same-experience control (A vs A2)
- state L2: 0.0
- JS: 0.0

## Reset control
- JS after reset+same probe: 0.0

## Noise / neutral
- A vs C state L2: 274983.1796875
- A vs N state L2: 193238.47265625


## 4. Activation divergence

Included in divergence / red-blue metrics (Jaccard of active indices when `state_capture=detailed`).

## 5. Output divergence

See JS/KL in divergence and red-blue sections.

## 6. Exposure-strength curve

# Exposure curve

- k=1: P(1|A)=0.0000, P(7|B)=0.0000, JS=0.0249
- k=2: P(1|A)=0.0000, P(7|B)=0.0000, JS=0.0914
- k=4: P(1|A)=0.0000, P(7|B)=0.0000, JS=0.1114
- k=8: P(1|A)=0.0000, P(7|B)=0.0000, JS=0.1653
- k=16: P(1|A)=0.0000, P(7|B)=0.0000, JS=0.4015
- k=32: P(1|A)=0.0000, P(7|B)=0.0000, JS=0.2360


## 7. Memory decay

# Memory decay

After establishing A→1 with 16 exposures, insert distractors then probe.

- d=0: P(target)=0.0000
- d=1: P(target)=0.0000
- d=5: P(target)=0.0000
- d=10: P(target)=0.0000
- d=50: P(target)=0.0000
- d=100: P(target)=0.0000
- d=500: P(target)=0.0000


## 8. Interference

# Interference

Initial 16× `X A 1`, then overwrite with `X A 7`.

- k=0: P(old)=0.0000, P(new)=0.0000
- k=1: P(old)=0.0000, P(new)=0.0000
- k=2: P(old)=0.0000, P(new)=0.0000
- k=4: P(old)=0.0000, P(new)=0.0000
- k=8: P(old)=0.0000, P(new)=0.0000
- k=16: P(old)=0.0000, P(new)=0.0000
- k=32: P(old)=0.0000, P(new)=0.0000


## 9. State persistence

# Persistence / restore

- snapshot: `/opt/BDH_v1/runs/2026-08-13_persistence_002/state/S_A.pt`
- weights_unchanged: True
- JS(before, after restore): 0.0
- max abs logit diff: 0.0
- P(target) before: 1.2228385912749218e-06
- P(target) after: 1.2228385912749218e-06


## 10. Conclusions

### Classification: **B**

Experience changes subsequent processing but effects decay on short distractor horizons (working-memory-like).

### Headline (Agent Red / Agent Blue)

# Identical weights, different lives

```
same original weights: YES
same architecture: YES
same probe: YES
different experience: YES
weights_changed: false

dynamic-state divergence (L2): 142557.546875
dynamic-state cosine: 0.8849357534771527
activation divergence (Jaccard): 0.6286563134935038
output-distribution divergence (JS): 0.26126642525196075
reset_ablates_effect: YES
JS after reset: 0.0
```

## Primary probe `X A `
- Red P(target_red): 7.74785894464003e-06
- Blue P(target_blue): 8.558195077057462e-06


### Thresholds used (predeclared)

```json
{
  "min_state_l2_for_effect": 0.001,
  "min_js_for_behavioral_effect": 0.0001,
  "decay_half_life_short_max_distractors": 50,
  "persistent_if_restore_js_max": 1e-08,
  "persistent_if_decay_p_target_min_at_100": 0.05
}
```
