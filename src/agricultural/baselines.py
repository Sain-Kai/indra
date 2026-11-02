"""Classical Machine Learning Baselines for Multi-Crop District Yield Prediction in India.

Implements:
1. State-Crop Empirical Mean Baseline
2. Random Forest Regressor
3. LightGBM Regressor (with Quantile Uncertainty intervals)
4. XGBoost Regressor

Evaluated using strict Geographical Holdout (unseen districts) across 13 crops and 31 states.
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb

from src.common.metrics import crop_yield_metrics, prediction_interval_coverage

logger = logging.getLogger(__name__)


class CropYieldBaselinePipeline:
    """Manages feature transformation, geographical holdout splitting, and baseline evaluations."""

    def __init__(
        self,
        cleaned_parquet_path: Union[str, Path] = "/home/neymar-jr/DSC Project/data_processed/upag_district_cleaned_2024_25.parquet",
        test_district_frac: float = 0.20,
        random_state: int = 42
    ):
        self.cleaned_parquet_path = Path(cleaned_parquet_path)
        self.test_district_frac = test_district_frac
        self.random_state = random_state
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    def load_and_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads clean UPAg district data and performs geographical holdout train/test split."""
        if not self.cleaned_parquet_path.exists():
            raise FileNotFoundError(f"Cleaned dataset not found at {self.cleaned_parquet_path}")

        df = pd.read_parquet(self.cleaned_parquet_path)

        # Unique districts for geographical holdout
        rng = np.random.RandomState(self.random_state)
        unique_districts = df["district"].unique()
        n_test = int(len(unique_districts) * self.test_district_frac)
        test_districts = set(rng.choice(unique_districts, size=n_test, replace=False))

        train_df = df[~df["district"].isin(test_districts)].copy().reset_index(drop=True)
        test_df = df[df["district"].isin(test_districts)].copy().reset_index(drop=True)

        return train_df, test_df

    def prepare_features(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Encodes features and extracts arrays for model training."""
        cat_cols = ["crop", "state", "season"]
        num_cols = ["area_thousand_ha"]

        # Fit encoder on train categoricals
        train_cat = self.encoder.fit_transform(train_df[cat_cols])
        test_cat = self.encoder.transform(test_df[cat_cols])

        # Target encoding: State-Crop mean yield from training set (proxy historical performance)
        state_crop_means = train_df.groupby(["state", "crop"])["yield_kg_per_ha"].mean().to_dict()
        overall_mean = train_df["yield_kg_per_ha"].mean()

        def get_prior(row):
            return state_crop_means.get((row["state"], row["crop"]), overall_mean)

        train_prior = train_df.apply(get_prior, axis=1).values.reshape(-1, 1)
        test_prior = test_df.apply(get_prior, axis=1).values.reshape(-1, 1)

        # Numerical arrays
        train_num = train_df[num_cols].values
        test_num = test_df[num_cols].values

        X_train = np.hstack([train_cat, train_prior, train_num])
        X_test = np.hstack([test_cat, test_prior, test_num])

        y_train = train_df["yield_kg_per_ha"].values
        y_test = test_df["yield_kg_per_ha"].values

        feature_names = cat_cols + ["state_crop_prior_yield"] + num_cols
        return X_train, y_train, X_test, y_test, feature_names


def run_all_baselines() -> Dict[str, Dict[str, float]]:
    """Trains and compares all classical baselines for crop yield prediction."""
    pipeline = CropYieldBaselinePipeline()
    train_df, test_df = pipeline.load_and_split()
    X_train, y_train, X_test, y_test, feat_names = pipeline.prepare_features(train_df, test_df)

    results = {}

    # 1. State-Crop Prior / Empirical Mean Baseline
    state_crop_means = train_df.groupby(["state", "crop"])["yield_kg_per_ha"].mean().to_dict()
    overall_mean = train_df["yield_kg_per_ha"].mean()
    pred_prior = test_df.apply(lambda r: state_crop_means.get((r["state"], r["crop"]), overall_mean), axis=1).values
    results["State-Crop Mean Prior"] = crop_yield_metrics(y_test, pred_prior)

    # 2. Random Forest Regressor
    rf = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    results["Random Forest"] = crop_yield_metrics(y_test, pred_rf)

    # 3. XGBoost Regressor
    xgb_reg = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1
    )
    xgb_reg.fit(X_train, y_train)
    pred_xgb = xgb_reg.predict(X_test)
    results["XGBoost"] = crop_yield_metrics(y_test, pred_xgb)

    # 4. LightGBM Regressor (with Quantile Uncertainty)
    lgb_reg = lgb.LGBMRegressor(
        n_estimators=250,
        max_depth=7,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_reg.fit(X_train, y_train)
    pred_lgb = lgb_reg.predict(X_test)
    results["LightGBM"] = crop_yield_metrics(y_test, pred_lgb)

    # Quantile models for prediction intervals (10% and 90% bounds)
    lgb_lower = lgb.LGBMRegressor(objective="quantile", alpha=0.10, n_estimators=150, verbose=-1, random_state=42)
    lgb_lower.fit(X_train, y_train)
    pred_lower = lgb_lower.predict(X_test)

    lgb_upper = lgb.LGBMRegressor(objective="quantile", alpha=0.90, n_estimators=150, verbose=-1, random_state=42)
    lgb_upper.fit(X_train, y_train)
    pred_upper = lgb_upper.predict(X_test)

    pi_res = prediction_interval_coverage(y_test, pred_lower, pred_upper)
    results["LightGBM 80% PI"] = pi_res

    return results, train_df, test_df


if __name__ == "__main__":
    results, train_df, test_df = run_all_baselines()

    print("=" * 85)
    print("TASK 1 — MULTI-CROP DISTRICT YIELD PREDICTION: BASELINES BENCHMARK")
    print(f"Data: UPAg 2024-25 Cleaned Panel (13 Crops, 31 States, 715 Districts)")
    print(f"Splits: Geographical Holdout | Train: {len(train_df)} rows | Test: {len(test_df)} rows (Unseen Districts)")
    print("=" * 85)
    print(f"{'Model':<24} | {'MAE (kg/ha)':<12} | {'RMSE (kg/ha)':<13} | {'R2':<8} | {'WAPE (%)':<10} | {'Pearson r':<10}")
    print("-" * 85)

    for name in ["State-Crop Mean Prior", "Random Forest", "XGBoost", "LightGBM"]:
        m = results[name]
        print(f"{name:<24} | {m['mae']:<12.1f} | {m['rmse']:<13.1f} | {m['r2']:<8.3f} | {m['wape_pct']:<10.1f} | {m['pearson_r']:<10.3f}")

    print("-" * 85)
    if "LightGBM 80% PI" in results:
        pi = results["LightGBM 80% PI"]
        print(f"Prediction Interval Evaluation (80% Nominal Target):")
        print(f"  - Actual Coverage (PICP): {pi['picp_pct']:.1f}%")
        print(f"  - Mean Interval Width (MPIW): {pi['mpiw']:.1f} kg/ha")
    print("=" * 85)
