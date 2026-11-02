"""Trains and benchmarks Deep Learning Sequence Models (BiGRU & Temporal Transformer)
on the North Indian Ocean Tropical Cyclone benchmark with uncertainty estimation.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure DSC Project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.common.metrics import cyclone_intensity_metrics, mean_track_error
from src.cyclone.baselines import CycloneXGBoostBaseline, evaluate_persistence_baseline
from src.cyclone.dataset import CycloneDatasetBuilder
from src.cyclone.models import CycloneBiGRUForecaster, CycloneTemporalTransformer, gaussian_nll_track_loss


def build_sequence_tensors(
    df: pd.DataFrame,
    seq_len: int = 4,
    horizons: List[int] = [6, 12, 24, 48, 72]
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], Dict[int, torch.Tensor], pd.DataFrame]:
    """Converts storm track table into sliding-window sequence tensors for RNN / Transformer."""
    feature_cols = CycloneXGBoostBaseline.FEATURE_COLS

    X_seqs = []
    targets_coords = {h: [] for h in horizons}
    targets_intensity = {h: [] for h in horizons}
    meta_rows = []

    for sid, storm in df.groupby("sid"):
        storm = storm.sort_values("datetime").reset_index(drop=True)
        n = len(storm)
        if n < seq_len:
            continue

        feats = storm[feature_cols].values

        for i in range(seq_len - 1, n):
            # Window of past observations
            window = feats[i - seq_len + 1: i + 1]
            X_seqs.append(window)
            meta_rows.append(storm.iloc[i].to_dict())

            for h in horizons:
                dlat = storm.loc[i, f"target_dlat_{h}h"]
                dlon = storm.loc[i, f"target_dlon_{h}h"]
                dwind = storm.loc[i, f"target_dwind_{h}h"]
                dpres = storm.loc[i, f"target_pres_{h}h"] - storm.loc[i, "pres_hpa"]
                valid = storm.loc[i, f"target_valid_{h}h"]

                targets_coords[h].append([dlat if valid else 0.0, dlon if valid else 0.0, valid])
                targets_intensity[h].append([dwind if valid else 0.0, dpres if valid else 0.0, valid])

    X_tensor = torch.tensor(np.array(X_seqs), dtype=torch.float32)
    tgt_coords_tensor = {h: torch.tensor(np.array(targets_coords[h]), dtype=torch.float32) for h in horizons}
    tgt_intensity_tensor = {h: torch.tensor(np.array(targets_intensity[h]), dtype=torch.float32) for h in horizons}
    meta_df = pd.DataFrame(meta_rows)

    return X_tensor, tgt_coords_tensor, tgt_intensity_tensor, meta_df


def train_eval_model(
    model: nn.Module,
    X_tr: torch.Tensor,
    tgt_c_tr: Dict[int, torch.Tensor],
    tgt_i_tr: Dict[int, torch.Tensor],
    X_val: torch.Tensor,
    tgt_c_val: Dict[int, torch.Tensor],
    tgt_i_val: Dict[int, torch.Tensor],
    X_te: torch.Tensor,
    meta_te: pd.DataFrame,
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cuda"
) -> Dict[int, Dict[str, float]]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    n_samples = len(X_tr)
    horizons = sorted(tgt_c_tr.keys())

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_samples)

        for b_start in range(0, n_samples, batch_size):
            idx = perm[b_start: b_start + batch_size]
            bx = X_tr[idx].to(device)

            optimizer.zero_grad()
            preds = model(bx)

            total_loss = 0.0
            for h in horizons:
                tc = tgt_c_tr[h][idx].to(device)
                ti = tgt_i_tr[h][idx].to(device)
                valid = tc[:, 2] == 1.0

                if not torch.any(valid):
                    continue

                # Gaussian NLL for coordinates (calibrated uncertainty)
                loss_c = gaussian_nll_track_loss(
                    preds[h]["d_coords"][valid],
                    tc[valid, :2],
                    preds[h]["log_var"][valid]
                )
                # Smooth L1 for intensity
                loss_i = F.smooth_l1_loss(preds[h]["d_intensity"][valid], ti[valid, :2])

                total_loss += loss_c + 0.1 * loss_i

            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

    # Evaluation on holdout test set
    model.eval()
    with torch.no_grad():
        preds_test = model(X_te.to(device))

    results = {}
    for h in horizons:
        valid_mask = meta_te[f"target_valid_{h}h"].values == 1
        if not np.any(valid_mask):
            continue

        pred_dlat = preds_test[h]["d_coords"][:, 0].cpu().numpy()[valid_mask]
        pred_dlon = preds_test[h]["d_coords"][:, 1].cpu().numpy()[valid_mask]
        pred_dwind = preds_test[h]["d_intensity"][:, 0].cpu().numpy()[valid_mask]
        pred_dpres = preds_test[h]["d_intensity"][:, 1].cpu().numpy()[valid_mask]

        sub = meta_te[valid_mask]
        pred_lat = sub["lat"].values + pred_dlat
        pred_lon = sub["lon"].values + pred_dlon
        pred_wind = np.clip(sub["wind_kt"].values + pred_dwind, 15, 175)
        pred_pres = np.clip(sub["pres_hpa"].values + pred_dpres, 880, 1020)

        true_lat = sub[f"target_lat_{h}h"].values
        true_lon = sub[f"target_lon_{h}h"].values
        true_wind = sub[f"target_wind_{h}h"].values
        true_pres = sub[f"target_pres_{h}h"].values

        mte = mean_track_error(pred_lat, pred_lon, true_lat, true_lon)
        wind_res = cyclone_intensity_metrics(pred_wind, true_wind)
        pres_res = cyclone_intensity_metrics(pred_pres, true_pres)

        # Track uncertainty (standard deviation in km)
        log_var = preds_test[h]["log_var"].cpu().numpy()[valid_mask]
        std_deg = np.sqrt(np.exp(log_var))
        # Approximate 1 degree ~ 111 km
        std_km = float(np.mean(std_deg) * 111.13)

        results[h] = {
            "mean_track_error_km": mte,
            "wind_mae_kt": wind_res["mae"],
            "pres_mae_hpa": pres_res["mae"],
            "mean_uncertainty_std_km": std_km,
            "count": int(np.sum(valid_mask))
        }

    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 85)
    print("TASK 2 (MODEL A) — DEEP LEARNING SPATIOTEMPORAL CYCLONE FORECASTING")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print("Strict Holdout Test Period: 2023-2024 Tropical Cyclones (Mocha, Biparjoy, Remal, Dana)")
    print("=" * 85)

    # 1. Dataset
    builder = CycloneDatasetBuilder()
    clean_df = builder.load_clean_data()
    feature_df = builder.extract_features_and_targets(clean_df)
    train_df, val_df, test_df = builder.split_by_event(feature_df, train_end_year=2020, val_end_year=2022)

    X_tr, tgt_c_tr, tgt_i_tr, _ = build_sequence_tensors(train_df, seq_len=4)
    X_val, tgt_c_val, tgt_i_val, _ = build_sequence_tensors(val_df, seq_len=4)
    X_te, tgt_c_te, tgt_i_te, meta_te = build_sequence_tensors(test_df, seq_len=4)

    print(f"Sequences: Train={len(X_tr)} | Val={len(X_val)} | Test={len(X_te)} (Sequence Length=4)")

    # 2. Train BiGRU
    print("\n--- Training Bidirectional GRU Sequence Model ---")
    t0 = time.time()
    bigru = CycloneBiGRUForecaster(input_dim=len(CycloneXGBoostBaseline.FEATURE_COLS), hidden_dim=128, num_layers=2)
    bigru_res = train_eval_model(bigru, X_tr, tgt_c_tr, tgt_i_tr, X_val, tgt_c_val, tgt_i_val, X_te, meta_te, epochs=30, lr=1e-3, device=device)
    print(f"BiGRU training finished in {time.time() - t0:.2f}s")

    # 3. Train Temporal Transformer
    print("\n--- Training Temporal Transformer Sequence Model ---")
    t0 = time.time()
    transformer = CycloneTemporalTransformer(input_dim=len(CycloneXGBoostBaseline.FEATURE_COLS), d_model=128, nhead=4, num_layers=2)
    transformer_res = train_eval_model(transformer, X_tr, tgt_c_tr, tgt_i_tr, X_val, tgt_c_val, tgt_i_val, X_te, meta_te, epochs=30, lr=8e-4, device=device)
    print(f"Temporal Transformer training finished in {time.time() - t0:.2f}s")

    # 4. Compare with Persistence and XGBoost
    pers_res = evaluate_persistence_baseline(test_df)
    xgb_baseline = CycloneXGBoostBaseline()
    xgb_baseline.fit(train_df, val_df)
    xgb_res = xgb_baseline.evaluate(test_df)

    # 5. Summary Table
    print("\n" + "=" * 105)
    print(f"{'Horizon':<10} | {'Model Architecture':<28} | {'Track Error (km)':<18} | {'Wind MAE (kt)':<15} | {'Cone Uncertainty':<18}")
    print("-" * 105)

    for h in [6, 12, 24, 48, 72]:
        if h in pers_res:
            p = pers_res[h]
            print(f"{f'+{h}h':<10} | {'Zero-Motion Persistence':<28} | {p['mean_track_error_km']:<18.1f} | {p['wind_mae_kt']:<15.1f} | {'N/A':<18}")
        if h in xgb_res:
            x = xgb_res[h]
            print(f"{'':<10} | {'XGBoost Multi-Horizon':<28} | {x['mean_track_error_km']:<18.1f} | {x['wind_mae_kt']:<15.1f} | {'Point Forecast':<18}")
        if h in bigru_res:
            b = bigru_res[h]
            unc_str = f"±{b['mean_uncertainty_std_km']:.1f} km"
            print(f"{'':<10} | {'Bidirectional GRU (Ours)':<28} | {b['mean_track_error_km']:<18.1f} | {b['wind_mae_kt']:<15.1f} | {unc_str:<18}")
        if h in transformer_res:
            t = transformer_res[h]
            unc_str = f"±{t['mean_uncertainty_std_km']:.1f} km"
            print(f"{'':<10} | {'Temporal Transformer (Ours)':<28} | {t['mean_track_error_km']:<18.1f} | {t['wind_mae_kt']:<15.1f} | {unc_str:<18}")
        print("-" * 105)


if __name__ == "__main__":
    main()
