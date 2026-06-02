"""PatchTST — patch-based Transformer for financial time series.

Key innovation over TFT: divides the sequence into *patches* (sub-sequences)
and applies attention over patches instead of individual time steps.

Advantages:
- O(P²) attention vs O(S²) for full-sequence transformers (P << S)
- Each patch captures local temporal structure before cross-patch attention
- Benchmarked as #1 architecture in 918-experiment study (arXiv:2603.16886)

Architecture (≈180K params):
    Input:  (B, seq_len=100, features=30)
    → Patch: 11 patches × 16 steps each, stride 8
    → Embed: Linear → (B, 11, d_model=128) + position encoding
    → Transformer Encoder × 3 layers, 8 heads
    → Output: [P(up), P(down), P(timeout)] or [P10, P50, P90]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


# ── Patch Embedding ──────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """Divide sequence into overlapping patches and project to d_model."""

    def __init__(self, seq_len: int, patch_len: int, stride: int,
                 num_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (seq_len - patch_len) // stride + 1

        # Project each patch (patch_len × num_features) → d_model
        self.projection = nn.Linear(patch_len * num_features, d_model)
        self.dropout = nn.Dropout(dropout)

        # Learnable position encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, seq_len, num_features) → (B, num_patches, d_model)"""
        B, S, F = x.shape
        patches = []
        for i in range(self.num_patches):
            start = i * self.stride
            patch = x[:, start:start + self.patch_len, :].reshape(B, -1)
            patches.append(patch)
        patches = torch.stack(patches, dim=1)  # (B, num_patches, patch_len*F)
        out = self.projection(patches)          # (B, num_patches, d_model)
        return self.dropout(out + self.pos_encoding)


# ── Transformer Encoder ──────────────────────────────────────────────

class TransformerEncoder(nn.Module):
    """Standard Transformer encoder with pre-LN and residual connections."""

    def __init__(self, d_model: int, num_heads: int = 8,
                 ff_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        ff_dim = ff_dim or d_model * 4

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with pre-norm
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = residual + x

        # FFN with pre-norm
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        return residual + x


# ── Full PatchTST Model ──────────────────────────────────────────────

class PatchTSTModel(nn.Module):
    """PatchTST for financial K-line prediction.

    Parameters
    ----------
    num_features : int
        Number of input features per time step.
    seq_len : int
        Number of historical time steps.
    patch_len : int
        Length of each patch (sub-sequence).
    stride : int
        Stride between patches (overlap when stride < patch_len).
    d_model : int
        Hidden dimension.
    num_heads : int
        Attention heads per encoder layer.
    num_layers : int
        Number of transformer encoder layers.
    dropout : float
        Dropout rate.
    output_mode : str
        'triple_barrier' → [P(up), P(down), P(timeout)]
        'quantile' → [P10, P50, P90] (regression-style)
    """

    def __init__(self,
                 num_features: int = 30,
                 seq_len: int = 100,
                 patch_len: int = 16,
                 stride: int = 8,
                 d_model: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 3,
                 dropout: float = 0.15,
                 output_mode: str = "triple_barrier"):
        super().__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_mode = output_mode

        self.embedding = PatchEmbedding(
            seq_len, patch_len, stride, num_features, d_model, dropout)

        self.encoders = nn.ModuleList([
            TransformerEncoder(d_model, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.norm_out = nn.LayerNorm(d_model)

        if output_mode == "triple_barrier":
            self.num_outputs = 3  # P(up), P(down), P(timeout)
        else:
            self.num_outputs = 3  # P10, P50, P90

        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, self.num_outputs),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Returns dict with:
          'logits' / 'quantiles' — raw outputs
          'probs' — softmax for triple_barrier mode
          'direction' — +1 (up), -1 (down), 0 (timeout/neutral)
          'confidence' — max probability among up/down
        """
        B = x.shape[0]

        # Patch embedding
        h = self.embedding(x)  # (B, num_patches, d_model)

        # Transformer encoders
        for encoder in self.encoders:
            h = encoder(h)

        # Global pooling (mean over patches)
        h = self.norm_out(h.mean(dim=1))  # (B, d_model)

        # Output
        raw = self.output_head(h)  # (B, num_outputs)

        if self.output_mode == "triple_barrier":
            probs = F.softmax(raw, dim=-1)  # [P(up), P(down), P(timeout)]
            p_up = probs[:, 0]
            p_down = probs[:, 1]

            direction = torch.where(
                p_up > p_down,
                torch.ones(B, device=x.device),
                torch.where(
                    p_down > p_up,
                    -torch.ones(B, device=x.device),
                    torch.zeros(B, device=x.device),
                ),
            )
            confidence = torch.max(p_up, p_down)

            return {
                "logits": raw,
                "probs": probs,
                "p_up": p_up,
                "p_down": p_down,
                "p_timeout": probs[:, 2],
                "direction": direction,
                "confidence": confidence,
            }
        else:
            return {
                "quantiles": raw,
                "p50": raw[:, 1],
                "direction": torch.sign(raw[:, 1]),
                "confidence": torch.sigmoid(torch.abs(raw[:, 1]) / (raw[:, 2] - raw[:, 0] + 1e-6)),
            }


# ── Loss Functions ────────────────────────────────────────────────────

def cross_entropy_loss(outputs: dict, y: torch.Tensor) -> torch.Tensor:
    """Cross-entropy for triple-barrier labels (y ∈ {0,1,2} = up/down/timeout)."""
    return F.cross_entropy(outputs["logits"], y.long())


def pinball_loss(y_pred: torch.Tensor, y_true: torch.Tensor,
                 quantiles: list[float] = [0.1, 0.5, 0.9]) -> torch.Tensor:
    """Pinball (quantile) loss."""
    errors = y_true.unsqueeze(1) - y_pred
    q = torch.tensor(quantiles, device=y_pred.device).unsqueeze(0)
    return torch.max(q * errors, (q - 1) * errors).mean()
