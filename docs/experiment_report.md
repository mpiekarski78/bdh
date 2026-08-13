# BDH Experience-Driven State Experiment

## Environment

- device: NVIDIA GB10
- torch: 2.13.0+cu130
- cuda: 13.0
- machine: aarch64
- commit: 2b0d7a45b058d4309c84a10e0768d541fe18bdc2
- fork: https://github.com/mpiekarski78/bdh

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
- probe: `my lo`
- targets: A→r  B→v

## Weight control
- any_weights_changed: **False**

## Final A vs B
- state L2: 114941.7890625
- state cosine: 0.7387937395288162
- output JS: 0.570887953042984
- P(target_a | A): 0.04322982206940651
- P(target_a | B): 0.010638287290930748
- P(target_b | B): 0.6073445081710815
- active Jaccard: 0.5891657424227126

## Same-experience control (A vs A2)
- state L2: 0.0
- JS: 0.0

## Reset control
- JS after reset+same probe: 0.0

## Noise / neutral
- A vs C state L2: 390455.48046875
- A vs N state L2: 199909.34375


## 4. Activation divergence

Included in divergence / red-blue metrics (Jaccard of active indices when `state_capture=detailed`).

## 5. Output divergence

See JS/KL in divergence and red-blue sections.

## 6. Exposure-strength curve

# Exposure curve

- k=1: P(1|A)=0.1571, P(7|B)=0.3699, JS=0.0400
- k=2: P(1|A)=0.0244, P(7|B)=0.8528, JS=0.3693
- k=4: P(1|A)=0.0010, P(7|B)=0.2585, JS=0.5039
- k=8: P(1|A)=0.0432, P(7|B)=0.6073, JS=0.5709
- k=16: P(1|A)=0.1329, P(7|B)=0.9228, JS=0.5747
- k=32: P(1|A)=0.0514, P(7|B)=0.0683, JS=0.5642


## 7. Memory decay

# Memory decay

After establishing A→1 with 16 exposures, insert distractors then probe.

- d=0: P(target)=0.1329
- d=1: P(target)=0.0785
- d=5: P(target)=0.1825
- d=10: P(target)=0.0024
- d=50: P(target)=0.8772
- d=100: P(target)=0.0000
- d=500: P(target)=0.9950


## 8. Interference

# Interference

Task `shakespeare_completion`: 16× `my lord` then overwrite with `my love`.

- k=0: P(old)=0.1329, P(new)=0.0237
- k=1: P(old)=0.4083, P(new)=0.4243
- k=2: P(old)=0.1109, P(new)=0.3238
- k=4: P(old)=0.5436, P(new)=0.1611
- k=8: P(old)=0.0141, P(new)=0.9496
- k=16: P(old)=0.4631, P(new)=0.4927
- k=32: P(old)=0.4220, P(new)=0.4600


## 9. State persistence

# Persistence / restore

- snapshot: `/opt/BDH_v1/runs/2026-08-13_persistence_003/state/S_A.pt`
- weights_unchanged: True
- JS(before, after restore): 0.0
- max abs logit diff: 0.0
- P(target) before: 0.13286076486110687
- P(target) after: 0.13286076486110687


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

dynamic-state divergence (L2): 315164.48046875
dynamic-state cosine: 0.4480532264342529
activation divergence (Jaccard): 0.4268108914308582
output-distribution divergence (JS): 0.006478470750153065
reset_ablates_effect: YES
JS after reset: 0.0
```

## Primary probe `my lo`
- Red P(target_red): 0.8751946687698364
- Blue P(target_blue): 0.06353792548179626


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
