"""Experiment metrics: state, activation, and output divergence."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def tensor_distances(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    fa = a.detach().float().reshape(-1)
    fb = b.detach().float().reshape(-1)
    l1 = float(torch.abs(fa - fb).sum().item())
    l2 = float(torch.linalg.vector_norm(fa - fb).item())
    n1 = float(torch.linalg.vector_norm(fa).item())
    n2 = float(torch.linalg.vector_norm(fb).item())
    cos = float(torch.dot(fa, fb).item() / (n1 * n2 + 1e-12))
    return {
        "l1": l1,
        "l2": l2,
        "cosine": cos,
        "relative_norm": float(l2 / (0.5 * (n1 + n2) + 1e-12)),
        "norm_a": n1,
        "norm_b": n2,
    }


def jaccard(a: set[int] | list[int], b: set[int] | list[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def activation_divergence(act_a: dict[str, Any], act_b: dict[str, Any]) -> dict[str, Any]:
    """Compare last-token activation logs from StatefulBDH."""
    out: dict[str, Any] = {"layers": []}
    if not act_a or not act_b:
        return {"error": "missing_activations", "layers": []}

    # Use final token entry
    la = act_a.get("layers", [])[-1]["per_layer"] if act_a.get("layers") else []
    lb = act_b.get("layers", [])[-1]["per_layer"] if act_b.get("layers") else []
    cosines = []
    jaccards = []
    for i, (ea, eb) in enumerate(zip(la, lb)):
        layer = {"layer": i}
        for key in ("x_sparse", "y_sparse", "xy_sparse"):
            # summary stats already stored; compare vector of summary fields
            va = torch.tensor([ea[key][k] for k in ("mean", "std", "max", "l2", "sparsity")])
            vb = torch.tensor([eb[key][k] for k in ("mean", "std", "max", "l2", "sparsity")])
            layer[f"{key}_summary_l2"] = float(torch.linalg.vector_norm(va - vb).item())
            layer[f"{key}_sparsity_a"] = ea[key]["sparsity"]
            layer[f"{key}_sparsity_b"] = eb[key]["sparsity"]
        for key in ("x_sparse_active", "y_sparse_active", "xy_sparse_active"):
            if key in ea and key in eb:
                j = jaccard(ea[key], eb[key])
                layer[key.replace("_active", "_jaccard")] = j
                jaccards.append(j)
        out["layers"].append(layer)
        # proxy cosine from xy sparsity summaries
        cosines.append(1.0 - min(1.0, layer.get("xy_sparse_summary_l2", 0.0)))
    out["mean_summary_proxy"] = float(np.mean(cosines)) if cosines else None
    out["mean_active_jaccard"] = float(np.mean(jaccards)) if jaccards else None
    return out


def output_divergence(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    *,
    top_k: int = 10,
    target_a: int | None = None,
    target_b: int | None = None,
) -> dict[str, float]:
    """Compare next-token distributions from last-position logits."""
    # Accept (B,T,V) or (V,) or (B,V)
    def _last(logits: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 3:
            return logits[0, -1]
        if logits.dim() == 2:
            return logits[0]
        return logits

    la = _last(logits_a).float()
    lb = _last(logits_b).float()
    pa = F.softmax(la, dim=-1)
    pb = F.softmax(lb, dim=-1)
    # KL(a||b) and KL(b||a), JS
    kl_ab = float(F.kl_div(pb.clamp_min(1e-12).log(), pa, reduction="sum").item())
    kl_ba = float(F.kl_div(pa.clamp_min(1e-12).log(), pb, reduction="sum").item())
    m = 0.5 * (pa + pb)
    js = 0.5 * float(F.kl_div(m.clamp_min(1e-12).log(), pa, reduction="sum").item()) + 0.5 * float(
        F.kl_div(m.clamp_min(1e-12).log(), pb, reduction="sum").item()
    )
    ka = torch.topk(pa, min(top_k, pa.numel())).indices.tolist()
    kb = torch.topk(pb, min(top_k, pb.numel())).indices.tolist()
    overlap = len(set(ka) & set(kb)) / float(top_k)

    # Spearman via rank correlation of probabilities
    ra = pa.argsort().argsort().float()
    rb = pb.argsort().argsort().float()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    spearman = float((ra * rb).sum().item() / ((ra.norm() * rb.norm()) + 1e-12))

    out = {
        "kl_ab": kl_ab,
        "kl_ba": kl_ba,
        "js": js,
        "top_k_overlap": overlap,
        "spearman": spearman,
        "confidence_a": float(pa.max().item()),
        "confidence_b": float(pb.max().item()),
        "argmax_a": int(pa.argmax().item()),
        "argmax_b": int(pb.argmax().item()),
    }
    if target_a is not None:
        out["p_target_a_on_a"] = float(pa[target_a].item())
        out["p_target_a_on_b"] = float(pb[target_a].item())
    if target_b is not None:
        out["p_target_b_on_a"] = float(pa[target_b].item())
        out["p_target_b_on_b"] = float(pb[target_b].item())
    return out


def association_strength(logits: torch.Tensor, target: int, foil: int | None = None) -> dict[str, float]:
    """P(target) and optional logit margin vs foil."""
    if logits.dim() == 3:
        logits = logits[0, -1]
    elif logits.dim() == 2:
        logits = logits[0]
    probs = F.softmax(logits.float(), dim=-1)
    out = {
        "p_target": float(probs[target].item()),
        "logit_target": float(logits[target].item()),
        "argmax": int(probs.argmax().item()),
        "confidence": float(probs.max().item()),
    }
    if foil is not None:
        out["p_foil"] = float(probs[foil].item())
        out["logit_margin"] = float((logits[target] - logits[foil]).item())
    return out
