"""Temporal Fusion Transformer for financial time series.

A simplified but faithful TFT implementation adapted for crypto K-line data:
- Variable Selection Network: automatic feature weighting per time step
- LSTM Encoder: sequence context encoding
- Multi-Head Interpretable Attention: learn which historical bars matter
- Quantile Outputs: [P10, P50, P90] for direction + uncertainty

Architecture (≈160K parameters — trainable on CPU, fast on GPU):

    Input: (B, seq_len=100, features=30)
        |
    [Variable Selection] → (B, seq, d_model=64)
        |
    [LSTM Encoder ×2]   → (B, seq, d_model)
        |
    [Multi-Head Attn ×4] → (B, seq, d_model) + attention weights
        |
    [Output Head]        → [P10, P50, P90] per step
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


# ── Variable Selection Network ───────────────────────────────────────

class VariableSelectionNetwork(nn.Module):
    """Per-timestep feature weighting via gated residual network.

    For each of the *num_features* input variables, learns a weight ∈ [0,1]
    that controls how much that variable contributes.  Weights are computed
    from both the feature value itself and a global context vector.
    """

    def __init__(self, input_dim: int, hidden_dim: int, context_dim: int = 0,
                 dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Per-feature weight network
        total_input = input_dim + context_dim
        self.weight_net = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Feature transformation (non-linear projection)
        self.transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (batch, seq_len, input_dim) → (batch, seq_len, hidden_dim)"""
        batch, seq, _ = x.shape

        # Flatten for per-element weighting
        x_flat = x.reshape(batch * seq, self.input_dim)

        if context is not None:
            ctx_flat = context.unsqueeze(1).expand(-1, seq, -1).reshape(batch * seq, -1)
            weight_input = torch.cat([x_flat, ctx_flat], dim=-1)
        else:
            weight_input = x_flat

        weights = self.weight_net(weight_input)  # (B*S, 1)
        transformed = self.transform(x_flat)      # (B*S, hidden_dim)

        return (weights * transformed).reshape(batch, seq, self.hidden_dim)


# ── Gated Residual Network ───────────────────────────────────────────

class GRN(nn.Module):
    """Gated Residual Network — TFT's building block.

    Two dense layers with ELU + gating + residual connection.
    The gate learns *how much* of the non-linear transform to use.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = F.elu(self.fc1(x))
        h = self.dropout(h)
        h = self.fc2(h)
        gate = torch.sigmoid(self.gate(x))
        return self.norm(residual + gate * h)


# ── Interpretable Multi-Head Attention ───────────────────────────────

class InterpretableMultiHeadAttention(nn.Module):
    """Multi-head attention that returns attention weights for interpretability.

    Uses a single set of shared values across heads (per TFT paper) so that
    attention weights can be meaningfully summed across heads.
    """

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, self.head_dim)  # shared V

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (output, attention_weights).

        x: (batch, seq_len, d_model)
        attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        batch, seq, _ = x.shape

        Q = self.q_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        scale = self.head_dim ** 0.5
        attn = (Q @ K.transpose(-2, -1)) / scale  # (B, H, S, S)

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ V  # (B, H, S, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch, seq, self.d_model)
        out = self.out_proj(out)

        return out, attn_weights


# ── Full TFT Model ───────────────────────────────────────────────────

class TFTModel(nn.Module):
    """Temporal Fusion Transformer for K-line sequence prediction.

    Parameters
    ----------
    num_features : int
        Number of input features (default 30).
    seq_len : int
        Sequence length — number of historical bars to process.
    d_model : int
        Hidden dimension throughout the model.
    num_heads : int
        Attention heads.
    lstm_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout rate for regularisation.
    quantiles : list[float]
        Quantiles to output (default [0.1, 0.5, 0.9]).
    """

    def __init__(self,
                 num_features: int = 30,
                 seq_len: int = 100,
                 d_model: int = 64,
                 num_heads: int = 4,
                 lstm_layers: int = 2,
                 dropout: float = 0.2,
                 quantiles: list[float] | None = None):
        super().__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.d_model = d_model
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        self.num_quantiles = len(self.quantiles)

        # Variable Selection — context_dim = d_model because we pass static_ctx
        self.vsn = VariableSelectionNetwork(num_features, d_model,
                                            context_dim=d_model, dropout=dropout)

        # LSTM Encoder
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Post-LSTM processing
        self.post_lstm_gate = nn.Linear(d_model, d_model)
        self.post_lstm_norm = nn.LayerNorm(d_model)

        # GRN after LSTM
        self.pre_attn_grn = GRN(d_model, dropout)

        # Multi-Head Attention
        self.attention = InterpretableMultiHeadAttention(d_model, num_heads, dropout)

        # Post-attention GRN
        self.post_attn_grn = GRN(d_model, dropout)

        # Output head: produce quantile predictions for the *last* time step
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.num_quantiles),
        )

        # Static enrichment: summary statistics of the whole sequence
        self.static_context = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : (batch, seq_len, num_features)

        Returns
        -------
        dict with keys:
            'quantiles' : (batch, num_quantiles) — predicted quantile values
            'attention' : (batch, num_heads, seq_len, seq_len) — attention weights
            'direction' : (batch,) — +1 for up, -1 for down (from P50)
            'confidence': (batch,) — signal strength in [0, 1]
            'uncertainty': (batch,) — P90-P10 width (larger = more uncertain)
        """
        batch, seq, _ = x.shape

        # ── Variable Selection ──
        # Use mean-pooled features over the sequence as static context
        static_ctx = self.static_context(x.mean(dim=1))  # (B, d_model)
        h = self.vsn(x, static_ctx)  # (B, S, d_model)

        # ── LSTM Encoder ──
        lstm_out, _ = self.lstm(h)  # (B, S, d_model)
        gate = torch.sigmoid(self.post_lstm_gate(h))
        h = self.post_lstm_norm(h + gate * lstm_out)

        # ── Attention ──
        h = self.pre_attn_grn(h)
        attn_out, attn_weights = self.attention(h)
        h = self.post_attn_grn(h + attn_out)

        # ── Output: use the last time step ──
        last_hidden = h[:, -1, :]  # (B, d_model)
        quantile_preds = self.output_head(last_hidden)  # (B, num_quantiles)

        # ── Derived outputs ──
        p10 = quantile_preds[:, 0]
        p50 = quantile_preds[:, 1]
        p90 = quantile_preds[:, 2]

        direction = torch.where(p50 > 0,
                                torch.ones_like(p50),
                                -torch.ones_like(p50))

        uncertainty = p90 - p10
        # Confidence: how far P50 is from zero, scaled by uncertainty
        confidence = torch.sigmoid(torch.abs(p50) / (uncertainty + 1e-6) - 0.5) * 2
        confidence = torch.clamp(confidence, 0.0, 1.0)

        return {
            'quantiles': quantile_preds,
            'direction': direction,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'attention': attn_weights,
            'p50': p50,
        }


# ── Loss Function ────────────────────────────────────────────────────

def pinball_loss(y_pred: torch.Tensor, y_true: torch.Tensor,
                 quantiles: list[float]) -> torch.Tensor:
    """Pinball (quantile) loss for multi-quantile prediction.

    y_pred: (batch, num_quantiles)
    y_true: (batch,)
    """
    errors = y_true.unsqueeze(1) - y_pred  # (B, Q)
    q = torch.tensor(quantiles, device=y_pred.device).unsqueeze(0)  # (1, Q)
    loss = torch.max(q * errors, (q - 1) * errors)
    return loss.mean()
