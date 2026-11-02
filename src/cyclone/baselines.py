"""Classical and Statistical Baseline Models for Tropical Cyclone Track and Intensity Forecasting.

Models:
1. Persistence Baseline (Zero-change assumption)
2. Kinematic Velocity Extrapolation (Constant velocity assumption)
3. Multi-Horizon Gradient Boosted Trees (XGBoost & LightGBM Regressors)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Ensure DSC Project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from src.common.metrics import (
    cyclone_intensity_metrics,
    haversine_distance,
    mean_track_error,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# STATISTICAL BASELINES
# ==============================================================================

def evaluate_persistence_baseline(
    test_df: pd.DataFrame,
    horizons: List[int] = [6, 12, 24, 48, 72]
) -> Dict[int, Dict[str, float]]:
    """Evaluates the zero-motion persistence baseline (storm remains at current coordinates)."""
    results = {}
    for h in horizons:
        valid = test_df[f"target_valid_{h}h"] == 1
        sub = test_df[valid]
        if len(sub) == 0:
            continue

        true_lat = sub[f"target_lat_{h}h"].values
        true_lon = sub[f"target_lon_{h}h"].values
        pred_lat = sub["lat"].values
        pred_lon = sub["lon"].values

        true_wind = sub[f"target_wind_{h}h"].values
        pred_wind = sub["wind_kt"].values
        true_pres = sub[f"target_pres_{h}h"].values
        pred_pres = sub["pres_hpa"].values

        mte = mean_track_error(pred_lat, pred_lon, true_lat, true_lon)
        wind_res = cyclone_intensity_metrics(pred_wind, true_wind)
        pres_res = cyclone_intensity_metrics(pred_pres, true_pres)

        results[h] = {
            "mean_track_error_km": mte,
            "wind_mae_kt": wind_res["mae"],
            "pres_mae_hpa": pres_res["mae"],
            "sample_count": len(sub),
        }
    return results


def evaluate_velocity_extrapolation_baseline(
    test_df: pd.DataFrame,
    horizons: List[int] = [6, 12, 24, 48, 72]
) -> Dict[int, Dict[str, float]]:
    """Extrapolates recent 6-hour translation velocity linearly forward in time."""
    results = {}
    for h in horizons:
        valid = test_df[f"target_valid_{h}h"] == 1
        sub = test_df[valid]
        if len(sub) == 0:
            continue

        # Linear projection: delta_lat = dlat_6h * (h / 6.0)
        scale = float(h) / 6.0
        pred_lat = sub["lat"].values + (sub["dlat_6h"].values * scale)
        pred_lon = sub["lon"].values + (sub["dlon_6h"].values * scale)

        true_lat = sub[f"target_lat_{h}h"].values
        true_lon = sub[f"target_lon_{h}h"].values

        # Intensity: persistence of recent 6h trend (capped)
        pred_wind = sub["wind_kt"].values + np.clip(sub["dwind_6h"].values * scale * 0.5, -30, 30)
        pred_pres = sub["pres_hpa"].values + np.clip(sub["dpres_6h"].values * scale * 0.5, -40, 40)

        true_wind = sub[f"target_wind_{h}h"].values
        true_pres = sub[f"target_pres_{h}h"].values

        mte = mean_track_error(pred_lat, pred_lon, true_lat, true_lon)
        wind_res = cyclone_intensity_metrics(pred_wind, true_wind)
        pres_res = cyclone_intensity_metrics(pred_pres, true_pres)

        results[h] = {
            "mean_track_error_km": mte,
            "wind_mae_kt": wind_res["mae"],
            "pres_mae_hpa": pres_res["mae"],
            "sample_count": len(sub),
        }
    return results


# ==============================================================================
# GRADIENT BOOSTED TREE (XGBOOST) BASELINE
# ==============================================================================

class CycloneXGBoostBaseline:
    """Multi-Horizon Gradient Boosted Tree forecaster for Tropical Cyclone Trajectory & Intensity."""

    FEATURE_COLS = [
        "lat", "lon", "wind_kt", "pres_hpa", "dist2land_km", "speed_kmh", "bearing_deg",
        "dlat_6h", "dlon_6h", "dlat_12h", "dlon_12h", "dlat_24h", "dlon_24h",
        "dwind_6h", "dwind_12h", "dwind_24h", "dpres_6h", "dpres_12h",
        "month", "sin_doy", "cos_doy"
    ]

    def __init__(self, horizons: List[int] = [6, 12, 24, 48, 72]):
        self.horizons = horizons
        self.models_dlat: Dict[int, xgb.XGBRegressor] = {}
        self.models_dlon: Dict[int, xgb.XGBRegressor] = {}
        self.models_dwind: Dict[int, xgb.XGBRegressor] = {}
        self.models_dpres: Dict[int, xgb.XGBRegressor] = {}

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None):
        """Train separate XGBoost models for each forecast horizon."""
        X_train = train_df[self.FEATURE_COLS].values

        for h in self.horizons:
            # Mask valid target rows
            mask_tr = train_df[f"target_valid_{h}h"] == 1
            if not np.any(mask_tr):
                continue

            X_tr = X_train[mask_tr]
            y_dlat_tr = train_df.loc[mask_tr, f"target_dlat_{h}h"].values
            y_dlon_tr = train_df.loc[mask_tr, f"target_dlon_{h}h"].values
            y_dwind_tr = train_df.loc[mask_tr, f"target_dwind_{h}h"].values
            y_dpres_tr = train_df.loc[mask_tr, f"target_pres_{h}h"].values - train_df.loc[mask_tr, "pres_hpa"].values

            # Hyperparameters tuned for trajectory regression
            params = {
                "n_estimators": 200,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "random_state": 42,
                "n_jobs": -1,
            }

            model_dlat = xgb.XGBRegressor(**params)
            model_dlat.fit(X_tr, y_dlat_tr)
            self.models_dlat[h] = model_dlat

            model_dlon = xgb.XGBRegressor(**params)
            model_dlon.fit(X_tr, y_dlon_tr)
            self.models_dlon[h] = model_dlon

            model_dwind = xgb.XGBRegressor(**params)
            model_dwind.fit(X_tr, y_dwind_tr)
            self.models_dwind[h] = model_dwind

            model_dpres = xgb.XGBRegressor(**params)
            model_dpres.fit(X_tr, y_dpres_tr)
            self.models_dpres[h] = model_dpres

    def evaluate(self, test_df: pd.DataFrame) -> Dict[int, Dict[str, float]]:
        """Evaluate multi-horizon performance on test holdout."""
        X_test = test_df[self.FEATURE_COLS].values
        results = {}

        for h in self.horizons:
            mask = test_df[f"target_valid_{h}h"] == 1
            if not np.any(mask) or h not in self.models_dlat:
                continue

            sub = test_df[mask]
            X_sub = X_test[mask]

            # Predict deltas
            pred_dlat = self.models_dlat[h].predict(X_sub)
            pred_dlon = self.models_dlon[h].predict(X_sub)
            pred_dwind = self.models_dwind[h].predict(X_sub)
            pred_dpres = self.models_dpres[h].predict(X_sub)

            pred_lat = sub["lat"].values + pred_dlat
            pred_lon = sub["lon"].values + pred_dlon
            pred_wind = np.clip(sub["wind_kt"].values + pred_dwind, 15, 170)
            pred_pres = np.clip(sub["pres_hpa"].values + pred_dpres, 880, 1020)

            true_lat = sub[f"target_lat_{h}h"].values
            true_lon = sub[f"target_lon_{h}h"].values
            true_wind = sub[f"target_wind_{h}h"].values
            true_pres = sub[f"target_pres_{h}h"].values

            mte = mean_track_error(pred_lat, pred_lon, true_lat, true_lon)
            wind_res = cyclone_intensity_metrics(pred_wind, true_wind)
            pres_res = cyclone_intensity_metrics(pred_pres, true_pres)

            results[h] = {
                "mean_track_error_km": mte,
                "wind_mae_kt": wind_res["mae"],
                "pres_mae_hpa": pres_res["mae"],
                "sample_count": len(sub),
            }
        return results
