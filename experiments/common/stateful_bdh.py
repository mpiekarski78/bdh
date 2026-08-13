"""Stateful BDH wrapper: recurrent ρ equivalent to public Attention.forward.

Public attention (bdh.Attention.forward):
    QR = RoPE(Q); KR = QR  # K is Q
    scores = (QR @ KR.mT).tril(diagonal=-1)
    return scores @ V

Recurrent form (per token t, after emitting y_t):
    y_t = QR_t @ ρ
    ρ += outer(QR_t, V_t)

ρ does not include the current token (tril diagonal=-1).
One ρ is maintained per layer. Upstream bdh.py is never modified.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

import bdh

CaptureLevel = Literal["none", "summary", "detailed"]


def _activation_summary(t: torch.Tensor) -> dict[str, float]:
    flat = t.detach().float().reshape(-1)
    nnz = int((flat > 0).sum().item())
    return {
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()) if flat.numel() else 0.0,
        "max": float(flat.max().item()) if flat.numel() else 0.0,
        "l2": float(torch.linalg.vector_norm(flat).item()),
        "sparsity": float(1.0 - nnz / flat.numel()) if flat.numel() else 0.0,
        "nnz": nnz,
        "numel": int(flat.numel()),
    }


def _top_active_indices(t: torch.Tensor, k: int = 64) -> list[int]:
    flat = t.detach().float().reshape(-1)
    if flat.numel() == 0:
        return []
    k = min(k, flat.numel())
    vals, idx = torch.topk(flat, k)
    # Keep only strictly positive activations when sparse-positive.
    keep = vals > 0
    return [int(i) for i, ok in zip(idx.tolist(), keep.tolist()) if ok]


@dataclass
class StateSnapshot:
    """Serializable dynamic state."""

    rho: list[torch.Tensor]  # per layer, CPU tensors
    position: int
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rho": self.rho,
            "position": self.position,
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "StateSnapshot":
        return StateSnapshot(rho=list(d["rho"]), position=int(d["position"]), meta=dict(d.get("meta") or {}))


class StatefulBDH(nn.Module):
    """Wraps an unmodified bdh.BDH with snapshot-able recurrent attention state."""

    def __init__(
        self,
        model: bdh.BDH,
        *,
        capture_level: CaptureLevel = "none",
        active_topk: int = 64,
    ):
        super().__init__()
        self.model = model
        self.config = model.config
        self.capture_level: CaptureLevel = capture_level
        self.active_topk = active_topk
        self.last_activations: dict[str, Any] = {}
        self._rho: list[torch.Tensor] | None = None
        self._position = 0
        self._batch_size: int | None = None

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _dims(self) -> tuple[int, int, int, int]:
        C = self.config
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh
        return C.n_layer, nh, N, D

    def reset_dynamic_state(self, batch_size: int = 1) -> None:
        n_layer, nh, N, D = self._dims()
        device = self.device
        dtype = next(self.model.parameters()).dtype
        self._rho = [
            torch.zeros(batch_size, nh, N, D, device=device, dtype=dtype)
            for _ in range(n_layer)
        ]
        self._position = 0
        self._batch_size = batch_size
        self.last_activations = {}

    def get_state_snapshot(self) -> StateSnapshot:
        if self._rho is None:
            self.reset_dynamic_state(1)
        assert self._rho is not None
        return StateSnapshot(
            rho=[r.detach().to("cpu").clone() for r in self._rho],
            position=self._position,
            meta={"batch_size": self._batch_size, "dtype": str(self._rho[0].dtype)},
        )

    def load_state_snapshot(self, snapshot: StateSnapshot | dict[str, Any]) -> None:
        if isinstance(snapshot, dict):
            snapshot = StateSnapshot.from_dict(snapshot)
        device = self.device
        dtype = next(self.model.parameters()).dtype
        self._rho = [r.to(device=device, dtype=dtype).clone() for r in snapshot.rho]
        self._position = int(snapshot.position)
        self._batch_size = int(snapshot.meta.get("batch_size", self._rho[0].shape[0]))

    def _ensure_state(self, batch_size: int) -> None:
        if self._rho is None or self._batch_size != batch_size:
            self.reset_dynamic_state(batch_size)

    def _rope_at(self, Q: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to Q at absolute positions. Q: (B, nh, T, N)."""
        attn = self.model.attn
        freqs = attn.freqs  # (1,1,1,N)
        # phases: (1,1,T,1) * freqs -> broadcast
        phases = positions.view(1, 1, -1, 1).to(dtype=freqs.dtype, device=freqs.device) * freqs
        return attn.rope(phases, Q)

    @torch.no_grad()
    def step(
        self,
        idx: torch.Tensor,
        *,
        return_activations: bool | None = None,
    ) -> torch.Tensor:
        """Process tokens left-to-right, updating ρ. Returns logits (B, T, vocab)."""
        self.model.eval()
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)
        B, T = idx.shape
        self._ensure_state(B)
        assert self._rho is not None

        C = self.config
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh
        capture: CaptureLevel
        if return_activations is None:
            capture = self.capture_level
        elif return_activations:
            capture = self.capture_level if self.capture_level != "none" else "summary"
        else:
            capture = "none"

        # Embed all new tokens; process sequentially for causal ρ updates.
        x_all = self.model.embed(idx).unsqueeze(1)  # B, 1, T, D
        x_all = self.model.ln(x_all)

        logits_steps: list[torch.Tensor] = []
        act_log: dict[str, Any] = {"layers": []}

        for t in range(T):
            x = x_all[:, :, t : t + 1, :]  # B, 1, 1, D
            pos = self._position
            pos_tensor = torch.tensor([pos], device=idx.device, dtype=self.model.attn.freqs.dtype)
            layer_acts: list[dict[str, Any]] = []

            for level in range(C.n_layer):
                x_latent = x @ self.model.encoder  # B, nh, 1, N
                x_sparse = F.relu(x_latent)

                QR = self._rope_at(x_sparse, pos_tensor)  # B, nh, 1, N
                # yKV = QR @ ρ  -> (B, nh, 1, D)
                yKV = torch.matmul(QR, self._rho[level])
                yKV = self.model.ln(yKV)

                # Update ρ after read (tril diagonal=-1)
                # outer(QR, V) with V = x broadcast over heads: (B, nh, N, D)
                # Use einsum (not QR.mT @ V) to avoid Triton outer-product kernels
                # that require python headers on some aarch64 installs.
                V = x.expand(B, nh, 1, D)  # B, nh, 1, D
                outer = torch.einsum("bhtn,bhtd->bhnd", QR, V)
                self._rho[level] = self._rho[level] + outer

                y_latent = yKV @ self.model.encoder_v
                y_sparse = F.relu(y_latent)
                xy_sparse = x_sparse * y_sparse
                # Dropout inactive in eval
                xy_sparse = self.model.drop(xy_sparse)

                yMLP = (
                    xy_sparse.transpose(1, 2).reshape(B, 1, 1, N * nh) @ self.model.decoder
                )
                y = self.model.ln(yMLP)
                x = self.model.ln(x + y)

                if capture != "none":
                    entry = {
                        "x_sparse": _activation_summary(x_sparse),
                        "y_sparse": _activation_summary(y_sparse),
                        "xy_sparse": _activation_summary(xy_sparse),
                        "yKV": _activation_summary(yKV),
                    }
                    if capture == "detailed":
                        entry["x_sparse_active"] = _top_active_indices(x_sparse, self.active_topk)
                        entry["y_sparse_active"] = _top_active_indices(y_sparse, self.active_topk)
                        entry["xy_sparse_active"] = _top_active_indices(xy_sparse, self.active_topk)
                    layer_acts.append(entry)

            logits_t = x.view(B, 1, D) @ self.model.lm_head
            logits_steps.append(logits_t)
            self._position += 1
            if capture != "none":
                act_log["layers"].append({"token_offset": t, "position": pos, "per_layer": layer_acts})

        if capture != "none":
            self.last_activations = act_log
        return torch.cat(logits_steps, dim=1)

    @torch.no_grad()
    def forward_full(self, idx: torch.Tensor) -> torch.Tensor:
        """Oracle: unmodified model forward on full sequence (no ρ persistence)."""
        self.model.eval()
        logits, _ = self.model(idx)
        return logits

    def save_snapshot(self, path: str | Any) -> None:
        snap = self.get_state_snapshot()
        torch.save(snap.to_dict(), path)

    def load_snapshot_file(self, path: str | Any) -> None:
        data = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_snapshot(data)


def rho_distance(a: StateSnapshot, b: StateSnapshot) -> dict[str, float]:
    """Aggregate distances between two ρ snapshots."""
    assert len(a.rho) == len(b.rho)
    total_l1 = 0.0
    total_l2 = 0.0
    dots = 0.0
    na = 0.0
    nb = 0.0
    per_layer = []
    for i, (ra, rb) in enumerate(zip(a.rho, b.rho)):
        fa = ra.float().reshape(-1)
        fb = rb.float().reshape(-1)
        l1 = float(torch.abs(fa - fb).sum().item())
        l2 = float(torch.linalg.vector_norm(fa - fb).item())
        d = float(torch.dot(fa, fb).item())
        n1 = float(torch.linalg.vector_norm(fa).item())
        n2 = float(torch.linalg.vector_norm(fb).item())
        cos = 1.0 if l2 == 0.0 else d / (n1 * n2 + 1e-12)
        rel = 0.0 if (n1 + n2) == 0 else l2 / (0.5 * (n1 + n2) + 1e-12)
        per_layer.append(
            {"layer": i, "l1": l1, "l2": l2, "cosine": cos, "relative_norm": rel, "norm_a": n1, "norm_b": n2}
        )
        total_l1 += l1
        total_l2 += l2
        dots += d
        na += n1 * n1
        nb += n2 * n2
    na = na ** 0.5
    nb = nb ** 0.5
    total_cos = 1.0 if total_l2 == 0.0 else float(dots / (na * nb + 1e-12))
    return {
        "l1": total_l1,
        "l2": total_l2,
        "cosine": total_cos,
        "relative_norm": float(0.0 if (na + nb) == 0 else total_l2 / (0.5 * (na + nb) + 1e-12)),
        "norm_a": float(na),
        "norm_b": float(nb),
        "position_a": a.position,
        "position_b": b.position,
        "per_layer": per_layer,
    }


def snapshots_allclose(a: StateSnapshot, b: StateSnapshot, atol: float = 1e-5) -> bool:
    if a.position != b.position or len(a.rho) != len(b.rho):
        return False
    for ra, rb in zip(a.rho, b.rho):
        if not torch.allclose(ra.float(), rb.float(), atol=atol, rtol=1e-5):
            return False
    return True
