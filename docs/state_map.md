# BDH State Map

Traced from the public Pathway implementation in [`bdh.py`](../bdh.py) (`BDH.forward` / `Attention.forward`), commit recorded in each run’s `environment.json`.

This document classifies tensors by role. Names follow the **code**, with paper ρ/σ noted where they correspond.

Default config (`BDHConfig`): `n_layer=6`, `n_embd=256`, `n_head=4`, `mlp_internal_dim_multiplier=128`, `vocab_size=256`, `dropout=0.1`.

Derived: `N = mlp_internal_dim_multiplier * n_embd / n_head = 8192`.

---

## Model weights (slow learned parameters)

These are `nn.Parameter` values trained by backprop. They **do not** change during `eval()` inference exposure in our experiments (verified by SHA256 checksums).

| Tensor | Shape | Dtype (typical) | Approx. size (fp32) | Init | Gradients | Survives across forwards |
|--------|-------|-----------------|---------------------|------|-----------|--------------------------|
| `embed.weight` | `(256, 256)` | fp32 | 0.25 MB | `Normal(0, 0.02)` | yes (train) | yes (fixed at infer) |
| `encoder` | `(4, 256, 8192)` | fp32 | 32 MB | `Normal(0, 0.02)` | yes | yes |
| `encoder_v` | `(4, 256, 8192)` | fp32 | 32 MB | `Normal(0, 0.02)` | yes | yes |
| `decoder` | `(32768, 256)` | fp32 | 32 MB | `Normal(0, 0.02)` | yes | yes |
| `lm_head` | `(256, 256)` | fp32 | 0.25 MB | `Normal(0, 0.02)` | yes | yes |

Notes:

- `nn.LayerNorm(..., elementwise_affine=False, bias=False)` → **no** affine parameters.
- Encoder/decoder/`encoder_v` are **shared across layers** (loop over `n_layer` reuses the same modules).

---

## Frozen non-trainable buffers

| Tensor | Shape | Role | Survives |
|--------|-------|------|----------|
| `attn.freqs` | `(1, 1, 1, N)` | RoPE frequency table (`torch.nn.Buffer`) | yes; not updated by experience |

---

## Dynamic recurrent / Hebbian working state (paper ρ)

The public `Attention.forward` is:

```python
scores = (QR @ KR.mT).tril(diagonal=-1)
return scores @ V
```

with `K is Q` and `KR = QR = RoPE(Q)`.

This is numerically equivalent to a recurrent accumulator **ρ**:

```text
y_t = QR_t @ ρ
ρ ← ρ + outer(QR_t, V_t)      # update AFTER read (diagonal=-1)
```

| State | Shape (per layer) | Dtype | Approx. size (fp32, B=1) | Init | Update location | Reset | Gradients | Survives between `BDH.forward` calls? |
|-------|-------------------|-------|--------------------------|------|-----------------|-------|-----------|----------------------------------------|
| `ρ[level]` | `(B, n_head, N, D)` = `(1, 4, 8192, 256)` | model dtype | ~32 MB / layer; **~192 MB** for 6 layers | zeros | `StatefulBDH.step` after attention read | `reset_dynamic_state()` | no (@torch.no_grad experiments) | **No** in upstream `bdh.py`; **Yes** in `StatefulBDH` |
| RoPE `position` | scalar int | int | tiny | 0 | increments per token in `step` | reset with state | no | only in wrapper |

**Upstream behavior:** each `BDH.forward(idx)` rebuilds attention from the provided token sequence only. There is **no** `nn.Module` buffer that carries ρ across separate Python calls. `generate()` re-encodes the full growing prefix every token (no KV/ρ cache).

**Experiment layer:** [`experiments/common/stateful_bdh.py`](../experiments/common/stateful_bdh.py) keeps one ρ per layer and exposes:

- `get_state_snapshot()` / `load_state_snapshot()`
- `reset_dynamic_state()`
- `step(tokens)`

Equivalence to full-sequence `BDH.forward` is tested in [`tests/test_equivalence.py`](../tests/test_equivalence.py).

---

## Temporary activations (per forward / per token)

Produced inside the layer loop; not retained unless instrumentation hooks capture them.

| Activation | Shape | Notes |
|------------|-------|-------|
| `x` (residual) | `(B, 1, T, D)` | token embeddings then layer residuals |
| `x_latent` | `(B, nh, T, N)` | `x @ encoder` |
| `x_sparse` | `(B, nh, T, N)` | `ReLU(x_latent)` — sparse positive |
| `yKV` | `(B, nh, T, D)` | attention output |
| `y_latent` / `y_sparse` | `(B, nh, T, N)` | value path through `encoder_v` + ReLU |
| `xy_sparse` | `(B, nh, T, N)` | `x_sparse * y_sparse` then dropout |
| `yMLP` / `y` | `(B, 1, T, D)` | decoded residual update |
| `logits` | `(B, T, vocab)` | `x @ lm_head` |

Instrumentation levels (`none` / `summary` / `detailed`) record magnitude, sparsity, and optionally top-k active indices — not full tensors by default.

---

## What changes during inference exposure?

| Kind | Changes during eval exposure? |
|------|-------------------------------|
| Slow weights | **No** (asserted via hash) |
| ρ / position (wrapper) | **Yes** — this is the experience substrate under test |
| Temporary activations | Yes, ephemerally |
| Dropout masks | Disabled in `eval()` for measurement runs |

---

## Memory summary (default config, B=1, fp32)

- Trainable weights ≈ 96.5 MB
- Dynamic ρ (6 layers) ≈ 192 MB
- Snapshots are stored as CPU clones via `torch.save`
