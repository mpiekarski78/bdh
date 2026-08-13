# BDH experience-driven state experiments

> **Fork** of [`pathwaycom/bdh`](https://github.com/pathwaycom/bdh) (Pathway’s public BDH-GPU baseline).  
> Upstream paper: [arXiv:2509.26507](https://doi.org/10.48550/arXiv.2509.26507).  
> This is **not** the official Pathway repository.

## What this fork adds

A measurement harness around **unmodified** upstream `bdh.py` / `train.py` to answer:

> Can two identical BDH models (same weights, same initial dynamic state) diverge internally and behaviorally after different experience histories **without** changing slow learned weights?

**Result (reported run): Category B — short-term adaptive memory.**

| Check | Outcome |
|-------|---------|
| Same architecture / same checkpoint | yes |
| Trainable weights after experience | unchanged (SHA256) |
| Dynamic state ρ after different histories | diverges (L2 ≫ 0) |
| Same probe, different output distributions | yes (v1 JS ≈ 0.26; v2 dedicated-task JS ≈ 0.57) |
| Reset ρ then re-probe | divergence disappears (JS → 0) |
| Snapshot → restore | exact match |

Interpretation: experience modifies inference-time state and subsequent processing, but the effect behaves like working memory (cleared by reset), not a durable long-term associative store.

### Experiment progression

| Phase | Status | Notes |
|-------|--------|-------|
| 1 Reproduce BDH | done | Full train on GB10; loss 5.65 → 0.11 |
| 2 State map | done | [`docs/state_map.md`](docs/state_map.md) |
| 3 Instrumentation | done | `StatefulBDH` ρ snapshot/reset/load |
| 4 Divergence suite | done | v1 synthetic bytes; v2 Shakespeare completions |
| 5 Red / Blue headline | done | Same weights, different lives |
| 5b Prior-relative association | done | B can raise P(v) vs prior; A does not raise P(r); 1 distractor byte can wipe it |
| 6 Cross-process persistence | partial | In-process restore is exact; longer-lived association still unproven |
| 7 Explicit memory store | not started | Out of scope until Category C/D |

Default probe task is now **`shakespeare_completion`** (`my lord` / `my love`). The original `symbol_association` protocol is kept for comparison (`--task symbol_association`).

Dated history: [`docs/CHANGELOG.md`](docs/CHANGELOG.md). Full report: [`docs/experiment_report.md`](docs/experiment_report.md).

### Layout

```text
bdh.py, train.py          # upstream, unmodified
experiments/              # harness + CLI runners
docs/                     # state map, protocol, report, hardware/metrics
datasets/synthetic/       # byte-level probe histories
tests/                    # recurrent ρ ≡ full-sequence attention
```

### Hardware (reported run)

| Item | Value |
|------|-------|
| GPU | NVIDIA GB10 (`aarch64`) |
| CUDA / PyTorch | 13.0 / 2.13.0+cu130 |
| Python | 3.12.3 |
| Baseline train | batch 32, block 512, 3000 iters, bf16 |
| Train loss | 5.65 → 0.11 (next-byte CE on Tiny Shakespeare) |

### Metrics

- **State (ρ):** L1, L2, cosine, relative norm  
- **Activations:** mean/std/max/L2/sparsity; Jaccard of top-k active indices  
- **Outputs:** KL, Jensen–Shannon, top-k overlap, Spearman ranks, confidence  
- **Controls:** weight hash, ρ reset, same-experience twins, length-matched noise/neutral

Full write-up:

- [`docs/hardware_and_metrics.md`](docs/hardware_and_metrics.md)
- [`docs/experiment_report.md`](docs/experiment_report.md)
- [`docs/experiment_protocol.md`](docs/experiment_protocol.md)
- [`docs/state_map.md`](docs/state_map.md)
- [`docs/CHANGELOG.md`](docs/CHANGELOG.md)
- [`docs/FORK.md`](docs/FORK.md)

### Quick start (this fork)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install numpy requests matplotlib

# train + save checkpoint (logged run dir under runs/)
python -m experiments.baseline.run_baseline --checkpoint checkpoints/base.pt

python tests/test_equivalence.py
python -m experiments.divergence.run_divergence --checkpoint checkpoints/base.pt --task shakespeare_completion
python -m experiments.report.run_red_blue --checkpoint checkpoints/base.pt
python -m experiments.report.run_report
```

Upstream’s original demo remains available as `python train.py` (does not save a checkpoint by itself).

---

# Upstream: Pathway BDH (Dragon Hatchling)

Content below is from the upstream project for context and attribution.  
Official repo: [github.com/pathwaycom/bdh](https://github.com/pathwaycom/bdh) · [Pathway](https://pathway.com)

## Bridging the Gap Between Transformers and the Brain

**BDH (Dragon Hatchling)** is a biologically inspired large language model architecture that connects principles of deep learning with the foundations of neuroscience. Developed by researchers at [Pathway](https://pathway.com), BDH provides a theoretical and practical framework for understanding the emergence of reasoning and generalization in artificial systems.

Paper:

> *A. Kosowski, P. Uznański, J. Chorowski, Z. Stamirowska, M. Bartoszkiewicz.*  
> [_The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain_](https://doi.org/10.48550/arXiv.2509.26507), arXiv (2025).

## Overview

BDH represents a **scale-free, locally interacting network of neurons** capable of intrinsic reasoning dynamics. BDH scales like a Transformer on performance benchmarks—yet retains full interpretability and theoretical grounding in the fine-grained dynamics of neuron interactions.

**Key properties:**

- **Scale-free network topology** mimicking biological connectivity
- **Locally interacting neuron particles** with excitatory/inhibitory dynamics
- **Hebbian working memory** based on synaptic plasticity, displaying monosemanticity
- **GPU-friendly state-space formulation** for efficient implementation
- **Interpretable activations** that are sparse and positive

Empirically, BDH matches **GPT-2–scale Transformers** across language and translation tasks at equivalent parameter scales (10M–1B).

## Architecture

<img src="figs/architecture.png" width="600"/>

## Relation to Transformers

<img src="figs/vocab.png" width="600"/>

BDH and the Transformer share attention-inspired computation; however, BDH’s graph-based architecture makes its attention **emerge naturally from neuron-level interactions**, reflecting attention as seen in biological systems.

## Scaling Laws

<img src="figs/bdh_scaling.png" width="600"/>

BDH follows **Transformer-like scaling laws**, maintaining parameter efficiency while achieving interpretability at any scale.

## Sudoku Benchmark (Pathway)

Note: The Sudoku Extreme result refers to Pathway’s **internal** BDH implementation, not to the current open-source baseline in this repository. See Pathway’s [Sudoku bench post](https://pathway.com/research/beyond-transformers-sudoku-bench).

| Model | Sudoku Extreme Accuracy | Relative Cost |
|------|------------------------|--------------|
| Pathway BDH (internal) | 97.4% | 10× lower, no chain-of-thought |
| Leading LLMs (O3-mini, DeepSeek R1, Claude 3.7 8K) | ~0% | High (chain-of-thought) |

## Upstream install / train

```bash
pip install -r requirements.txt
python train.py
```

## Learn and discuss (upstream)

- [SuperDataScience podcast](https://www.youtube.com/watch?v=mfV44-mtg7c) with Adrian Kosowski  
- Coverage: [Forbes](https://www.forbes.com/sites/victordey/2025/10/08/can-ai-learn-and-evolve-like-a-brain-pathways-bold-research-thinks-so/), [Semafor](https://www.semafor.com/article/10/01/2025/new-ai-research-claims-to-be-getting-closer-to-modeling-human-brain), and others  
- Paper hubs: [Hugging Face](https://huggingface.co/papers/2509.26507), [Alphaxiv](https://alphaxiv.org/abs/2509.26507), [EmergentMind](https://emergentmind.com/papers/2509.26507)

## Community ports (upstream list)

- [adamskrodzki/bdh](https://github.com/adamskrodzki/bdh): dynamic vocabulary, stateful attention
- [mosure/burn_dragon_hatchling](https://github.com/mosure/burn_dragon_hatchling): Burn port
- [severian42/bdh](https://github.com/severian42/bdh): MLX port
- [Git-Faisal/bdh](https://github.com/Git-Faisal/bdh)
- [GrahLnn/bdh](https://github.com/GrahLnn/bdh)

## Acknowledgements (upstream)

Upstream thanks Andrej Karpathy for [nanoGPT](https://github.com/karpathy/nanoGPT/) and the tiny Shakespeare dataset used in the demo.
