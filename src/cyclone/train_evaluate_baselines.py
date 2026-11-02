"""Train and benchmark all cyclone baselines on the North Indian Ocean test set (2023-2024)."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure DSC Project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.cyclone.baselines import (
    CycloneXGBoostBaseline,
    evaluate_persistence_baseline,
    evaluate_velocity_extrapolation_baseline,
)
from src.cyclone.dataset import CycloneDatasetBuilder


def main():
    print("=" * 80)
    print("TASK 2 (MODEL A) — CYCLONE TRAJECTORY & INTENSITY FORECASTING BENCHMARK")
    print("Ground-truth: IBTrACS North Indian Ocean (1842-2026)")
    print("Holdout Test Event Period: 2023-2024 (Mocha, Biparjoy, Remal, Dana, etc.)")
    print("=" * 80)

    # 1. Build dataset
    builder = CycloneDatasetBuilder()
    clean_df = builder.load_clean_data()
    feature_df = builder.extract_features_and_targets(clean_df)
    train_df, val_df, test_df = builder.split_by_event(feature_df, train_end_year=2020, val_end_year=2022)

    print(f"\n[Data Splits] Train (<=2020): {len(train_df)} obs | Val (2021-2022): {len(val_df)} obs | Test (2023+): {len(test_df)} obs")

    # 2. Evaluate Persistence
    print("\n--- 1. Evaluating Zero-Motion Persistence Baseline ---")
    pers_res = evaluate_persistence_baseline(test_df)

    # 3. Evaluate Kinematic Extrapolation
    print("--- 2. Evaluating Linear Velocity Extrapolation Baseline ---")
    extrap_res = evaluate_velocity_extrapolation_baseline(test_df)

    # 4. Train & Evaluate XGBoost
    print("--- 3. Training & Evaluating Multi-Horizon XGBoost Forecaster ---")
    xgb_baseline = CycloneXGBoostBaseline()
    xgb_baseline.fit(train_df, val_df)
    xgb_res = xgb_baseline.evaluate(test_df)

    # 5. Format Comparison Table
    print("\n" + "=" * 90)
    print(f"{'Horizon':<10} | {'Model':<25} | {'Track Error (km)':<18} | {'Wind MAE (kt)':<15} | {'Pres MAE (hPa)':<15}")
    print("-" * 90)

    horizons = [6, 12, 24, 48, 72]
    for h in horizons:
        if h in pers_res:
            p = pers_res[h]
            print(f"{f'+{h}h':<10} | {'Persistence':<25} | {p['mean_track_error_km']:<18.1f} | {p['wind_mae_kt']:<15.1f} | {p['pres_mae_hpa']:<15.1f}")
        if h in extrap_res:
            e = extrap_res[h]
            print(f"{'':<10} | {'Velocity Extrapolation':<25} | {e['mean_track_error_km']:<18.1f} | {e['wind_mae_kt']:<15.1f} | {e['pres_mae_hpa']:<15.1f}")
        if h in xgb_res:
            x = xgb_res[h]
            print(f"{'':<10} | {'XGBoost Regressor':<25} | {x['mean_track_error_km']:<18.1f} | {x['wind_mae_kt']:<15.1f} | {x['pres_mae_hpa']:<15.1f}")
        print("-" * 90)


if __name__ == "__main__":
    main()
