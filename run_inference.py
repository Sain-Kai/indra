"""Unified Local Inference Runner for Crop Yield, Cyclone Forecasting & Coastal Disasters.

Uses lightweight CPU runtime engines (ONNX Runtime, LightGBM, XGBoost)
with ZERO heavy CUDA training dependencies.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure DSC Project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd

from src.agricultural.early_season_anomaly import EarlySeasonYieldEngine
from src.cyclone.baselines import CycloneXGBoostBaseline
from src.cyclone.dataset import CycloneDatasetBuilder
from src.disaster.exposure import CoastalExposureEngine


def run_cyclone_forecast(storm_id: str = "2024145N15088"):
    """Runs multi-horizon track & intensity forecast with calibrated uncertainty cones."""
    print("=" * 80)
    print("TASK 2 — CYCLONE TRAJECTORY & INTENSITY INFERENCE")
    print("=" * 80)

    model_file = PROJECT_ROOT / "models_exported" / "cyclone_xgboost_models.joblib"
    if model_file.exists():
        model: CycloneXGBoostBaseline = joblib.load(model_file)
    else:
        builder = CycloneDatasetBuilder()
        clean_df = builder.load_clean_data()
        feature_df = builder.extract_features_and_targets(clean_df)
        train_df, _, _ = builder.split_by_event(feature_df)
        model = CycloneXGBoostBaseline()
        model.fit(train_df)

    builder = CycloneDatasetBuilder()
    clean_df = builder.load_clean_data()
    feature_df = builder.extract_features_and_targets(clean_df)

    # Filter sample storm
    storm_pts = feature_df[feature_df["sid"] == storm_id]
    if storm_pts.empty:
        storm_pts = feature_df.iloc[-10:]
        sample_name = storm_pts["name"].iloc[0] if "name" in storm_pts else "Recent Storm"
        print(f"Using test storm: {sample_name} (SID: {storm_pts['sid'].iloc[0]})")
    else:
        sample_name = storm_pts["name"].iloc[0]
        print(f"Found storm: {sample_name} (SID: {storm_id})")

    curr_point = storm_pts.iloc[-1]
    curr_lat = float(curr_point["lat"])
    curr_lon = float(curr_point["lon"])
    curr_wind = float(curr_point["wind_kt"])
    curr_pres = float(curr_point["pres_hpa"])

    # Uncertainty radii from empirical validation (90% confidence cone radius)
    uncertainty_cone_radii_km = {6: 48.0, 12: 94.0, 24: 187.0, 48: 404.0, 72: 620.0}

    print(f"\n[Current Fix] Lat: {curr_lat:.2f}°N | Lon: {curr_lon:.2f}°E | Wind: {curr_wind:.1f} kt | Pres: {curr_pres:.1f} hPa")
    print("-" * 90)
    print(f"{'Horizon':<9} | {'Forecast Lat':<14} | {'Forecast Lon':<14} | {'Wind (kt)':<11} | {'Pres (hPa)':<12} | {'90% Cone Radius':<16}")
    print("-" * 90)

    forecast_pts = []
    X_sample = curr_point[CycloneXGBoostBaseline.FEATURE_COLS].values.reshape(1, -1)
    for h in [6, 12, 24, 48, 72]:
        if h in model.models_dlat:
            pred_dlat = float(model.models_dlat[h].predict(X_sample)[0])
            pred_dlon = float(model.models_dlon[h].predict(X_sample)[0])
            pred_dwind = float(model.models_dwind[h].predict(X_sample)[0])
            pred_dpres = float(model.models_dpres[h].predict(X_sample)[0])

            f_lat = round(curr_lat + pred_dlat, 2)
            f_lon = round(curr_lon + pred_dlon, 2)
            f_wind = round(max(15.0, curr_wind + pred_dwind), 1)
            f_pres = round(min(1020.0, curr_pres + pred_dpres), 1)
            cone_r = uncertainty_cone_radii_km.get(h, 100.0)

            forecast_pts.append({"horizon_h": h, "lat": f_lat, "lon": f_lon, "wind_kt": f_wind, "pres_hpa": f_pres, "cone_radius_km": cone_r})
            print(f"{f'+{h}h':<9} | {f_lat:<14.2f} | {f_lon:<14.2f} | {f_wind:<11.1f} | {f_pres:<12.1f} | ± {cone_r:<12.1f} km")
    print("=" * 90)

    # Trigger Coastal Exposure Assessment
    print("\nTASK 2 — COASTAL EXPOSURE & AGRICULTURAL RISK INTERSECTION")
    print("-" * 90)
    exposure_engine = CoastalExposureEngine()
    exposure_res = exposure_engine.assess_storm_impact(forecast_pts)
    summary = exposure_res["summary"]
    print(f"Projected Peak Storm Surge: {summary['projected_storm_surge_m']} meters")
    print(f"Total Coastal Cropland Exposed: {summary['total_exposed_cropland_ha']:,} hectares across {summary['total_coastal_districts_affected']} districts")
    if exposure_res["exposed_districts"]:
        top_dist = exposure_res["exposed_districts"][0]
        print(f"Highest Priority District: {top_dist['district']} ({top_dist['state']}) — Threat: {top_dist['threat_level']}, Cropland: {top_dist['exposed_cropland_ha']:,} ha")
    print("=" * 90)


def run_crop_yield_prediction(
    state: str = "Punjab",
    district: str = "Ludhiana",
    crop: str = "Wheat",
    season: str = "Rabi",
    growth_stage: str = "Flowering / Reproductive (60 DAS)"
):
    """Runs district-level crop yield prediction with uncertainty bounds & anomaly status."""
    print("=" * 80)
    print("TASK 1 — MULTI-CROP DISTRICT YIELD & ANOMALY INFERENCE")
    print("=" * 80)

    engine = EarlySeasonYieldEngine()
    res = engine.predict_yield(
        state=state,
        district=district,
        crop=crop,
        season=season,
        growth_stage=growth_stage
    )

    print(f"\nQuery Parameters:")
    print(f"  • Geography:    {res['query']['district']}, {res['query']['state']}")
    print(f"  • Crop & Cycle: {res['query']['crop']} ({res['query']['season']})")
    print(f"  • Growth Stage: {res['query']['growth_stage']}")
    print(f"\nModel Projections:")
    print(f"  ★ Point Forecast:           {res['predicted_yield_kg_ha']:,.1f} kg/ha")
    print(f"  ★ 80% Prediction Interval:  [{res['prediction_interval_80']['lower_kg_ha']:,.1f}  —  {res['prediction_interval_80']['upper_kg_ha']:,.1f} kg/ha]")
    print(f"  ★ 95% Confidence Interval:  [{res['prediction_interval_95']['lower_kg_ha']:,.1f}  —  {res['prediction_interval_95']['upper_kg_ha']:,.1f} kg/ha]")
    print(f"  ★ Historical District Norm: {res['historical_prior_kg_ha']:,.1f} kg/ha")
    print(f"  ★ Anomaly Status:           {res['anomaly']['category']} ({res['anomaly']['deviation_pct']:+.1f}%)")

    if "prithvi_foundation_model" in res and res["prithvi_foundation_model"].get("status") == "ACTIVE & FUSED":
        pm = res["prithvi_foundation_model"]
        print(f"\nMultimodal Satellite Foundation Model (Kaggle-Trained Prithvi-EO-2.0):")
        print(f"  ★ Model Checkpoint:         {pm['model_name']} ({pm['checkpoint']})")
        print(f"  ★ Status:                   {pm['status']}")
        print(f"  ★ Satellite Canopy Health:  {pm['satellite_vegetation_health_index']} (NDVI proxy)")
        print(f"  ★ ONNX CPU Latency:         {pm['inference_latency_ms']} ms")
        print(f"  ★ Spectral Optical Bands:   {', '.join(pm['optical_bands_processed'])}")

    print(f"\nAgro-Advisory:")
    print(f"  ★ {res['anomaly']['advisory']}")
    print("=" * 80)


def run_prithvi_onnx_inference():
    """Runs local inference using the Kaggle-trained exported ONNX model."""
    print("=" * 80)
    print("PRITHVI-EO-2.0 FOUNDATION MODEL (ONNX INFERENCE)")
    print("=" * 80)

    model_path = PROJECT_ROOT / "models_exported" / "prithvi_model.onnx"
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return

    session = ort.InferenceSession(str(model_path))
    in_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name

    # Dummy Sentinel-2 6-band optical tile (B, 6, 3, 224, 224)
    dummy_tile = np.random.randn(1, 6, 3, 224, 224).astype(np.float32)
    pred = session.run([out_name], {in_name: dummy_tile})[0]

    print(f"Loaded: {model_path.name} ({model_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"Input: 6 spectral optical bands across 3 temporal steps (224x224)")
    print(f"Predicted Prithvi Geospatial Value: {pred[0][0]:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run local inference for Crop Yield, Cyclones and Disasters")
    parser.add_argument("--task", choices=["cyclone", "crop", "prithvi", "all"], default="all")
    args = parser.parse_args()

    if args.task in ["cyclone", "all"]:
        run_cyclone_forecast()
    if args.task in ["crop", "all"]:
        run_crop_yield_prediction()
    if args.task in ["prithvi", "all"]:
        run_prithvi_onnx_inference()
