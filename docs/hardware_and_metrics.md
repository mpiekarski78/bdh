# Hardware and metrics

## Hardware / software (reported run)

| Item | Value |
|------|-------|
| Machine | DGX Spark-class host, `aarch64` |
| OS kernel | Linux 6.17.0-1014-nvidia |
| GPU | NVIDIA GB10 (1×) |
| CUDA | 13.0 |
| cuDNN | 92000 |
| Python | 3.12.3 |
| PyTorch | 2.13.0+cu130 |
| Training dtype | bfloat16 |
| `torch.compile` | disabled for baseline (`--no-compile`) |
| Upstream commit | `2b0d7a45b058d4309c84a10e0768d541fe18bdc2` (`pathwaycom/bdh`) |

## Baseline training (Phase 1)

Upstream-equivalent recipe after GPU was free:

| Hyperparameter | Value |
|----------------|-------|
| Dataset | Tiny Shakespeare (byte-level, vocab 256) |
| `BDHConfig` | `n_layer=6`, `n_embd=256`, `n_head=4`, `mlp_internal_dim_multiplier=128` |
| Batch size | 32 |
| Block size | 512 |
| Max iters | 3000 |
| Optimizer | AdamW, lr `1e-3`, weight decay `0.1` |
| Seed | 1337 |
| Final logged loss | **0.110** (from 5.65 at step 0) |
| Checkpoint | `checkpoints/base.pt` (not shipped in git; regenerate locally) |

Earlier constrained run (while other large GPU jobs were resident): batch 2 / block 128 / 500 iters, loss 5.67→2.19. Headline results below use the **full** 32×512×3000 checkpoint.

## Metrics used

### Dynamic state (ρ)

Per-layer recurrent attention state equivalent to public `Attention.forward`:

- L1 / L2 distance
- Cosine similarity
- Relative norm difference
- Tracked over experience steps and for final A vs B / Red vs Blue

### Activations (probe)

For sparse positive tensors (`x_sparse`, `y_sparse`, `xy_sparse`):

- mean, std, max, L2, sparsity
- Jaccard overlap of top-k active indices (`state_capture=detailed`)

### Output distributions (probe next-byte)

- KL divergence (both directions)
- Jensen–Shannon divergence
- Top-k overlap
- Spearman rank correlation of probability ranks
- Confidence / argmax
- Optional P(target) for synthetic association targets

### Controls

- SHA256 hash of all trainable parameters before vs after experience (`weights_changed`)
- Reset ρ then re-probe
- Identical experience on two clones (determinism floor)
- Length-matched neutral / noise histories

### Classification thresholds (predeclared)

See `experiments/report/run_report.py` and [`experiment_protocol.md`](experiment_protocol.md). Reported result: **Category B** (short-term adaptive memory).

## Headline numbers (full checkpoint)

### v1 — `symbol_association` (`X A ` → 1 vs 7)

Out of distribution for Tiny Shakespeare. Output distributions diverged, but P(target bytes) stayed ~1e-5.

| Quantity | Value |
|----------|-------|
| Weights changed | false |
| Dynamic-state L2 (Red/Blue) | ≈ 1.43×10⁵ |
| Output JS | ≈ 0.261 |
| JS after ρ reset | 0.0 |

### v2 — `shakespeare_completion` (`my lo` → r vs v)

In-distribution. Dedicated A vs B (8× `my lord` vs 8× `my love`):

| Quantity | Value |
|----------|-------|
| Weights changed | false |
| State L2 | ≈ 1.15×10⁵ |
| State cosine | ≈ 0.74 |
| Output JS | ≈ 0.571 |
| P(`v` \| B) | ≈ 0.61 |
| P(`r` \| A) | ≈ 0.043 (pretrained prior for “lord” is already high; extra `my lord` does not simply raise it) |
| Same-experience L2 / JS | 0 |
| JS after ρ reset | 0.0 |
| Restore | exact |

Red/Blue mixed lives (lord/love **and** Hamlet/Friends) dilute the specific `my lo` probe (JS ≈ 0.006) while still producing large ρ divergence (L2 ≈ 3.15×10⁵, cosine ≈ 0.45). Decay vs further Shakespeare context is **non-monotonic** — not a smooth forgetting curve.

Full narrative: [`experiment_report.md`](experiment_report.md).
