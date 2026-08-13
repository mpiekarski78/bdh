# Experiment Protocol

## Question

Can two identical BDH models (same architecture, same trained weights, same initial ρ) develop measurably different internal state and behavior after different experience histories, without changing slow weights?

## Stack

- Upstream: [pathwaycom/bdh](https://github.com/pathwaycom/bdh) (unmodified `bdh.py`)
- Fork: [mpiekarski78/bdh](https://github.com/mpiekarski78/bdh)
- Wrapper: `experiments.common.stateful_bdh.StatefulBDH` (recurrent ρ ≡ public attention)

## Predeclared classification thresholds

Used by `python -m experiments.report.run_report` **before** interpreting results:

| ID | Meaning | Default |
|----|---------|---------|
| A | State only / negligible behavioral effect | state L2 < 1e-3 or JS < 1e-4 |
| B | Short-term adaptive memory | behavioral effect + reset ablates + decays by ~50 distractors |
| C | Persistent associative state | restore works + slower decay |
| D | Candidate durable associative substrate | C + lasting useful differentiation |

Exact constants: see `THRESHOLDS` in `experiments/report/run_report.py`.

## Commands

```bash
source .venv/bin/activate

# Phase 1
python -m experiments.baseline.run_baseline --checkpoint checkpoints/base.pt

# Equivalence gate
python tests/test_equivalence.py

# Phase 4–5
python -m experiments.determinism.run_determinism --checkpoint checkpoints/base.pt
python -m experiments.divergence.run_divergence --checkpoint checkpoints/base.pt --task shakespeare_completion
python -m experiments.exposure.run_exposure_curve --checkpoint checkpoints/base.pt
python -m experiments.decay.run_decay --checkpoint checkpoints/base.pt --exposures 16
python -m experiments.interference.run_interference --checkpoint checkpoints/base.pt
python -m experiments.persistence.run_restore --checkpoint checkpoints/base.pt
python -m experiments.prior.run_prior --checkpoint checkpoints/base.pt
python -m experiments.report.run_report
```

Default `--task` is `shakespeare_completion`. Pass `--task symbol_association` to reproduce the original out-of-distribution byte protocol.


## Controls (required)

1. **Reset** ρ then re-probe — behavioral gap should shrink.
2. **Weight checksum** before/after — must be identical.
3. **Same experience** (A vs A2) — divergence ≈ determinism floor.
4. **Noise / neutral** length-matched histories — vs structured association.

## Hardware notes (reported workspace)

- Hardware: NVIDIA GB10 (`aarch64`), CUDA 13.0, PyTorch `2.13.0+cu130`, Python 3.12.3
- Full baseline (GPU free): batch 32, block 512, 3000 iters, bf16 → loss 5.65→0.11
- Earlier constrained smoke run (other large GPU jobs resident): batch 2 / block 128 / 500 iters
- Metric catalog and headline numbers: [`hardware_and_metrics.md`](hardware_and_metrics.md)
- Fork notes: [`FORK.md`](FORK.md)

## Run artifacts

Each run writes `runs/<date>_<name>_<nnn>/` with `config.json`, `environment.json`, `metrics.json`, `summary.md`, optional `plots/`, `state/`.
