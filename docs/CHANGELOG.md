# Changelog

Dated experiment and harness changes. The README keeps only current status plus a short progression table.

## 2026-08-13

- Forked `pathwaycom/bdh` as [`mpiekarski78/bdh`](https://github.com/mpiekarski78/bdh). Upstream `bdh.py` / `train.py` unmodified.
- Added recurrent ρ wrapper, run logging, and CLI experiments (divergence, exposure, decay, interference, restore, Red/Blue).
- Trained baseline on NVIDIA GB10 (CUDA 13.0, PyTorch 2.13.0+cu130): batch 32, block 512, 3000 steps, loss 5.65 → 0.11.
- **v1 probes (`symbol_association`):** `X A 1` vs `X A 7`. State diverged and output JS ≈ 0.26 on the full checkpoint, but P(target bytes) stayed ~1e-5 (out of distribution for a Shakespeare byte model). Classification **B**.
- README restructured so this fork’s harness leads; Pathway content is under an upstream heading.
- **v2 probes (`shakespeare_completion`, default):** in-distribution `my lord` vs `my love`, probe `my lo` → `r`/`v`. Dedicated A vs B: output JS ≈ **0.57**, P(`v`\|B) ≈ **0.61**, weights unchanged, reset clears the gap. Mixed Red/Blue lives still diverge in ρ but dilute the specific probe (JS ≈ 0.006). Decay is non-monotonic under further Shakespeare context. Classification remains **B**.
- **Prior-relative association:** empty prior P(`r`)=0.64, P(`v`)=0.18 on probe `my lo`. Repeating `my love` can raise P(`v`) vs prior (peak Δ ≈ +0.74 at 16 exposures); repeating `my lord` does **not** raise P(`r`) (Δ ≈ −0.5). Length-matched filler also suppresses P(`r`), so much of the lord drop is “any prefix,” not a learned lord association. Clean-filler forgetting is fragile (1 extra byte can collapse P(`v`)). Reset after B matches empty prior (JS=0). Still Category **B**.
- **Deliverable:** [`conclusion.md`](conclusion.md) — Category B closed. ρ is session residue, not a fact store. Phases 6–7 not pursued on this evidence.

Older `symbol_association` remains available via `--task symbol_association`.
