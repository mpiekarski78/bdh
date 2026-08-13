# Changelog

Dated experiment and harness changes. The README keeps only current status plus a short progression table.

## 2026-08-13

- Forked `pathwaycom/bdh` as [`mpiekarski78/bdh`](https://github.com/mpiekarski78/bdh). Upstream `bdh.py` / `train.py` unmodified.
- Added recurrent ρ wrapper, run logging, and CLI experiments (divergence, exposure, decay, interference, restore, Red/Blue).
- Trained baseline on NVIDIA GB10 (CUDA 13.0, PyTorch 2.13.0+cu130): batch 32, block 512, 3000 steps, loss 5.65 → 0.11.
- **v1 probes (`symbol_association`):** `X A 1` vs `X A 7`. State diverged and output JS ≈ 0.26 on the full checkpoint, but P(target bytes) stayed ~1e-5 (out of distribution for a Shakespeare byte model). Classification **B**.
- README restructured so this fork’s harness leads; Pathway content is under an upstream heading.
- **v2 probes (`shakespeare_completion`, default):** in-distribution `my lord` vs `my love`, probe `my lo` → `r`/`v`. Dedicated A vs B: output JS ≈ **0.57**, P(`v`\|B) ≈ **0.61**, weights unchanged, reset clears the gap. Mixed Red/Blue lives still diverge in ρ but dilute the specific probe (JS ≈ 0.006). Decay is non-monotonic under further Shakespeare context. Classification remains **B**.
- Audit: recurrent ρ matches concatenated `BDH.forward` (tiny model ~2e-7; trained 6-layer ckpt ~2e-5). Split history→probe matches concat. Weight hashes unchanged after `step`. Tiny Shakespeare is ASCII, so prior UTF-8 encoding did not affect reported runs; encoding now uses latin-1 (byte ids). Report no longer treats JS/L2 of `0.0` as missing via Python `or`.

Older `symbol_association` remains available via `--task symbol_association`.
