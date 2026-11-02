"""Deep Learning Sequence Architectures for Tropical Cyclone Trajectory and Intensity Forecasting.

Models:
1. CycloneBiGRUForecaster: Bidirectional GRU with Multi-Horizon Prediction Heads & Uncertainty Estimation
2. CycloneTemporalTransformer: Self-Attention Spatiotemporal Transformer for Trajectory Forecasting
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# BIDIRECTIONAL GRU TRAJECTORY & INTENSITY FORECASTER
# ==============================================================================

class CycloneBiGRUForecaster(nn.Module):
    """Bidirectional GRU trajectory forecaster with multi-horizon uncertainty heads.

    Input sequence: (Batch, Seq_Len, Input_Dim) representing chronological observations.
    Outputs: Dict mapping horizon h -> {
        'coords_delta': (Batch, 2) [d_lat, d_lon],
        'intensity_delta': (Batch, 2) [d_wind, d_pres],
        'track_uncertainty': (Batch, 2) [log_var_lat, log_var_lon]
    }
    """

    def __init__(
        self,
        input_dim: int = 21,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        horizons: List[int] = [6, 12, 24, 48, 72]
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.horizons = sorted(horizons)

        # Feature projection
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Recurrent backbone
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Multi-horizon heads
        rnn_out_dim = hidden_dim * 2
        self.horizon_heads = nn.ModuleDict()

        for h in self.horizons:
            self.horizon_heads[str(h)] = nn.Sequential(
                nn.Linear(rnn_out_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                # Output: 2 coords delta (lat, lon), 2 intensity delta (wind, pres), 2 log variances
                nn.Linear(hidden_dim, 6)
            )

    def forward(self, x: torch.Tensor) -> Dict[int, Dict[str, torch.Tensor]]:
        # x: (B, T, input_dim)
        feat = self.in_proj(x)
        out, _ = self.gru(feat)
        # Use final sequence representation
        last_rep = out[:, -1, :]  # (B, 2 * hidden_dim)

        outputs = {}
        for h in self.horizons:
            pred = self.horizon_heads[str(h)](last_rep)  # (B, 6)
            outputs[h] = {
                "d_coords": pred[:, 0:2],        # [dlat, dlon]
                "d_intensity": pred[:, 2:4],     # [dwind, dpres]
                "log_var": pred[:, 4:6]          # [log_var_lat, log_var_lon] for uncertainty ellipse
            }
        return outputs


# ==============================================================================
# TEMPORAL TRANSFORMER TRAJECTORY FORECASTER
# ==============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for track time steps."""

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class CycloneTemporalTransformer(nn.Module):
    """Spatiotemporal self-attention transformer for cyclone trajectory forecasting."""

    def __init__(
        self,
        input_dim: int = 21,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        horizons: List[int] = [6, 12, 24, 48, 72]
    ):
        super().__init__()
        self.horizons = sorted(horizons)
        self.in_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.heads = nn.ModuleDict()
        for h in self.horizons:
            self.heads[str(h)] = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 6)
            )

    def forward(self, x: torch.Tensor) -> Dict[int, Dict[str, torch.Tensor]]:
        # x: (B, T, input_dim)
        x_proj = self.pos_encoder(self.in_proj(x))
        enc_out = self.transformer(x_proj)
        last_rep = enc_out[:, -1, :]  # (B, d_model)

        outputs = {}
        for h in self.horizons:
            pred = self.heads[str(h)](last_rep)
            outputs[h] = {
                "d_coords": pred[:, 0:2],
                "d_intensity": pred[:, 2:4],
                "log_var": pred[:, 4:6]
            }
        return outputs


# ==============================================================================
# HETEROSCEDASTIC GAUSSIAN NLL LOSS (WITH UNCERTAINTY CALIBRATION)
# ==============================================================================

def gaussian_nll_track_loss(
    pred_delta: torch.Tensor,
    true_delta: torch.Tensor,
    log_var: torch.Tensor
) -> torch.Tensor:
    """Negative Log-Likelihood loss for calibrated trajectory cone of uncertainty.

    Loss = 0.5 * exp(-log_var) * (pred - true)^2 + 0.5 * log_var
    """
    precision = torch.exp(-torch.clamp(log_var, -5.0, 5.0))
    sq_err = (pred_delta - true_delta) ** 2
    loss = 0.5 * (precision * sq_err + log_var)
    return loss.mean()
