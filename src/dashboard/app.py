"""Web Dashboard Server for Indian Agricultural Intelligence & Coastal Disaster Forecasting.

Provides:
- Native Python standard library HTTP server with multi-threaded request handling (zero external dependencies).
- REST API for Crop Yield prediction, Early-season anomalies, and Soil health profiles.
- REST API for Tropical Cyclone multi-horizon forecasting (+6h to +72h) with uncertainty cones.
- REST API for Coastal storm surge, DEM elevation inundation tiers, and agricultural cropland exposure.
- Interactive dual-view dashboard frontend with Leaflet.js mapping.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any, Dict, List, Optional

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from src.agricultural.early_season_anomaly import EarlySeasonYieldEngine, GROWTH_STAGES
from src.cyclone.baselines import CycloneXGBoostBaseline
from src.cyclone.dataset import CycloneDatasetBuilder
from src.disaster.exposure import CoastalExposureEngine, COASTAL_DISTRICTS
from src.disaster.dem_processor import SRTMProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("INDRA_Dashboard")

# Global singleton engines
crop_engine: Optional[EarlySeasonYieldEngine] = None
cyclone_model: Optional[CycloneXGBoostBaseline] = None
exposure_engine: Optional[CoastalExposureEngine] = None
cached_storms: Dict[str, List[Dict[str, Any]]] = {}
metadata_cache: Dict[str, Any] = {}


def query_ollama_advisory(crop: str, district: str, state: str, stage: str, pred_yield: float, anom_cat: str, dev_pct: float, veg_idx: float) -> Optional[str]:
    """Queries local Ollama LLM (qwen2.5:0.5b) for dynamic agronomic advisory."""
    try:
        import urllib.request
        prompt = (
            f"You are an agricultural AI advisory expert for India. Crop: {crop}, District: {district}, {state}. "
            f"Growth Stage: {stage}. Predicted Yield: {pred_yield} kg/ha ({anom_cat}, {dev_pct:+.1f}%). "
            f"Satellite Vegetation Index: {veg_idx}. In 2 concise sentences, provide practical agronomic advice on irrigation, fertilizer, or pest management."
        )
        req_data = json.dumps({"model": "qwen2.5:0.5b", "prompt": prompt, "stream": False}).encode("utf-8")
        req = urllib.request.Request("http://localhost:11434/api/generate", data=req_data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "").strip()
    except Exception:
        return None


def query_ollama_cyclone_bulletin(storm_name: str, max_wind: float, min_pres: float, surge_m: float, affected_dists: int, exposed_ha: float) -> Optional[str]:
    """Queries local Ollama LLM (qwen2.5:0.5b) for disaster early-warning bulletin."""
    try:
        import urllib.request
        prompt = (
            f"You are a disaster emergency officer in India. Storm: {storm_name}. Max Sustained Wind: {max_wind} kt. "
            f"Central Pressure: {min_pres} hPa. Projected Storm Surge: {surge_m} m. "
            f"Affected Coastal Districts: {affected_dists}. Total Cropland at Risk: {exposed_ha:,.0f} hectares. "
            f"In 2 urgent sentences, provide an emergency disaster warning bulletin and evacuation/agricultural advisory."
        )
        req_data = json.dumps({"model": "qwen2.5:0.5b", "prompt": prompt, "stream": False}).encode("utf-8")
        req = urllib.request.Request("http://localhost:11434/api/generate", data=req_data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "").strip()
    except Exception:
        return None


def initialize_engines():
    """Initializes models and cached metadata on server launch."""
    global crop_engine, cyclone_model, exposure_engine, cached_storms, metadata_cache

    logger.info("Initializing EarlySeasonYieldEngine...")
    crop_engine = EarlySeasonYieldEngine()
    crop_engine.load_or_fit()

    logger.info("Loading Cyclone XGBoost multi-horizon models...")
    model_path = PROJECT_ROOT / "models_exported" / "cyclone_xgboost_models.joblib"
    if model_path.exists():
        cyclone_model = joblib.load(model_path)
    else:
        builder = CycloneDatasetBuilder()
        clean_df = builder.load_clean_data()
        feature_df = builder.extract_features_and_targets(clean_df)
        train_df, _, _ = builder.split_by_event(feature_df)
        cyclone_model = CycloneXGBoostBaseline()
        cyclone_model.fit(train_df)
        joblib.dump(cyclone_model, model_path)

    logger.info("Initializing CoastalExposureEngine and SRTM DEM processor...")
    exposure_engine = CoastalExposureEngine()

    # Pre-cache named cyclone events from IBTrACS
    logger.info("Caching historical cyclone storms...")
    builder = CycloneDatasetBuilder()
    clean_df = builder.load_clean_data()
    feat_df = builder.extract_features_and_targets(clean_df)

    named_sids = [
        ("DANA", "2024295N15092", 2024),
        ("REMAL", "2024145N14087", 2024),
        ("BIPARJOY", "2023156N10067", 2023),
        ("MOCHA", "2023129N08091", 2023),
        ("MICHAUNG", "2023334N08088", 2023),
        ("FENGAL", "2024329N04089", 2024),
    ]

    for name, sid, year in named_sids:
        sub = feat_df[feat_df["sid"] == sid]
        if not sub.empty:
            time_col = "datetime" if "datetime" in sub.columns else ("time" if "time" in sub.columns else None)
            if time_col:
                sub = sub.sort_values(time_col)
            records = sub.to_dict(orient="records")
            # Convert timestamps to string
            for r in records:
                for k, v in list(r.items()):
                    if isinstance(v, (pd.Timestamp, np.datetime64)):
                        r[k] = str(v)
                r["time"] = str(r.get("datetime", r.get("time", "")))
            cached_storms[name] = {
                "name": name,
                "sid": sid,
                "year": year,
                "records": records
            }

    # Pre-cache Agricultural metadata
    logger.info("Caching agricultural states, districts, crops, and seasons...")
    upag_file = PROJECT_ROOT / "data_processed" / "upag_district_cleaned_2024_25.parquet"
    if upag_file.exists():
        upag_df = pd.read_parquet(upag_file)
        states = sorted(upag_df["state"].unique().tolist())
        state_districts = {
            st: sorted(upag_df[upag_df["state"] == st]["district"].unique().tolist())
            for st in states
        }
        crops = sorted(upag_df["crop"].unique().tolist())
        seasons = sorted(upag_df["season"].unique().tolist())
    else:
        states = ["Punjab", "Haryana", "Uttar Pradesh", "Maharashtra", "Gujarat", "Andhra Pradesh", "Tamil Nadu", "Odisha", "West Bengal"]
        state_districts = {s: ["District-1", "District-2"] for s in states}
        crops = ["Wheat", "Rice", "Cotton", "Maize", "Gram", "Sugarcane"]
        seasons = ["Kharif", "Rabi", "Summer"]

    metadata_cache = {
        "states": states,
        "state_districts": state_districts,
        "crops": crops,
        "seasons": seasons,
        "growth_stages": list(GROWTH_STAGES.keys()),
        "storms": [
            {"name": name, "sid": s["sid"], "year": s["year"], "count": len(s["records"])}
            for name, s in cached_storms.items()
        ],
        "coastal_districts": [d["district"] for d in COASTAL_DISTRICTS]
    }
    logger.info("All engines and caches successfully initialized.")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP Server for fast concurrent API request handling."""
    daemon_threads = True


class DashboardRequestHandler(SimpleHTTPRequestHandler):
    """Handles REST API routing and static frontend serving."""

    def __init__(self, *args, **kwargs):
        directory = str(Path(__file__).resolve().parent)
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/":
            self.path = "/index.html"
            return super().do_GET()

        elif path == "/api/metadata":
            self.send_json_response(metadata_cache)

        elif path == "/api/status":
            onnx_exists = (PROJECT_ROOT / "models_exported" / "prithvi_model.onnx").exists()
            self.send_json_response({
                "status": "HEALTHY",
                "version": "1.0.0",
                "prithvi_foundation_model": "AVAILABLE (ONNX Int8/Float32)" if onnx_exists else "OFFLINE",
                "cyclone_model": "LOADED (XGBoost Multi-Horizon)",
                "crop_model": "LOADED (LightGBM Quantile PI)",
                "dem_raster": "ONLINE (SRTM 90m WGS84)",
                "architecture": "Option 2 Hybrid (Local CPU Inference + Cloud Kaggle Training)"
            })

        elif path == "/api/storm_track":
            query = urllib.parse.parse_qs(parsed.query)
            storm_name = query.get("name", ["DANA"])[0].upper()
            if storm_name in cached_storms:
                self.send_json_response(cached_storms[storm_name])
            else:
                self.send_error(404, f"Storm '{storm_name}' not found.")

        elif path == "/api/reliability_metrics":
            from src.evaluation.benchmark_suite import evaluate_crop_yield_reliability, evaluate_cyclone_reliability
            if not hasattr(DashboardRequestHandler, "_cached_reliability"):
                DashboardRequestHandler._cached_reliability = {
                    "agricultural": evaluate_crop_yield_reliability(),
                    "cyclone": evaluate_cyclone_reliability()
                }
            self.send_json_response(DashboardRequestHandler._cached_reliability)

        else:
            return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8")
        try:
            payload = json.loads(body) if body else {}
        except Exception as e:
            self.send_error(400, f"Invalid JSON payload: {e}")
            return

        if path == "/api/predict_crop":
            try:
                state = payload.get("state", "Punjab")
                district = payload.get("district", "Ludhiana")
                crop = payload.get("crop", "Wheat")
                season = payload.get("season", "Rabi")
                growth_stage = payload.get("growth_stage", "Pre-Harvest Maturity (120+ DAS)")
                area = float(payload.get("area_thousand_ha", 25.0))

                result = crop_engine.predict_yield(
                    state=state,
                    district=district,
                    crop=crop,
                    season=season,
                    area_thousand_ha=area,
                    growth_stage=growth_stage
                )

                # Query local LLM (qwen2.5:0.5b GGUF via Ollama) for generative advisory
                veg_idx = result.get("prithvi_foundation_model", {}).get("satellite_vegetation_health_index", 0.75)
                llm_text = query_ollama_advisory(
                    crop=crop, district=district, state=state, stage=growth_stage,
                    pred_yield=result["predicted_yield_kg_ha"],
                    anom_cat=result["anomaly"]["category"],
                    dev_pct=result["anomaly"]["deviation_pct"],
                    veg_idx=veg_idx
                )
                if llm_text:
                    result["anomaly"]["advisory"] = llm_text
                    result["anomaly"]["advisory_source"] = "Ollama LLM (qwen2.5:0.5b GGUF Neural Engine)"
                else:
                    result["anomaly"]["advisory_source"] = "Agro-Meteorological Expert Rule Engine"

                self.send_json_response(result)
            except Exception as e:
                logger.error(f"Error in /api/predict_crop: {e}", exc_info=True)
                self.send_json_response({"error": str(e)}, status=500)

        elif path == "/api/predict_cyclone":
            try:
                # May receive fix parameters directly or a storm name
                storm_name = payload.get("storm_name")
                if storm_name and storm_name.upper() in cached_storms:
                    storm_data = cached_storms[storm_name.upper()]
                    records = storm_data["records"]
                    curr_fix = records[-1]
                    hist_points = [
                        {"lat": r["lat"], "lon": r["lon"], "wind_kt": r["wind_kt"], "pres_hpa": r["pres_hpa"], "time": r["time"]}
                        for r in records
                    ]
                else:
                    curr_fix = {
                        "lat": float(payload.get("lat", 17.5)),
                        "lon": float(payload.get("lon", 88.5)),
                        "wind_kt": float(payload.get("wind_kt", 60.0)),
                        "pres_hpa": float(payload.get("pres_hpa", 985.0)),
                        "dist2land_km": float(payload.get("dist2land_km", 250.0)),
                        "speed_kmh": float(payload.get("speed_kmh", 18.0)),
                        "bearing_deg": float(payload.get("bearing_deg", 340.0)),
                        "dlat_6h": float(payload.get("dlat_6h", 0.6)),
                        "dlon_6h": float(payload.get("dlon_6h", -0.2)),
                        "dlat_12h": float(payload.get("dlat_12h", 1.1)),
                        "dlon_12h": float(payload.get("dlon_12h", -0.4)),
                        "dlat_24h": float(payload.get("dlat_24h", 2.2)),
                        "dlon_24h": float(payload.get("dlon_24h", -0.8)),
                        "dwind_6h": float(payload.get("dwind_6h", 5.0)),
                        "dwind_12h": float(payload.get("dwind_12h", 10.0)),
                        "dwind_24h": float(payload.get("dwind_24h", 15.0)),
                        "dpres_6h": float(payload.get("dpres_6h", -4.0)),
                        "dpres_12h": float(payload.get("dpres_12h", -8.0)),
                        "month": 10,
                        "sin_doy": 0.8,
                        "cos_doy": 0.5
                    }
                    hist_points = [
                        {"lat": curr_fix["lat"] - 1.2, "lon": curr_fix["lon"] + 0.5, "wind_kt": curr_fix["wind_kt"] - 15, "pres_hpa": curr_fix["pres_hpa"] + 10, "time": "T-12h"},
                        {"lat": curr_fix["lat"] - 0.6, "lon": curr_fix["lon"] + 0.2, "wind_kt": curr_fix["wind_kt"] - 5, "pres_hpa": curr_fix["pres_hpa"] + 4, "time": "T-6h"},
                        {"lat": curr_fix["lat"], "lon": curr_fix["lon"], "wind_kt": curr_fix["wind_kt"], "pres_hpa": curr_fix["pres_hpa"], "time": "Fix (Current)"}
                    ]

                # Prepare feature vector for XGBoost
                feat_vals = [
                    curr_fix.get("lat", 17.5),
                    curr_fix.get("lon", 88.5),
                    curr_fix.get("wind_kt", 60.0),
                    curr_fix.get("pres_hpa", 985.0),
                    curr_fix.get("dist2land_km", 200.0),
                    curr_fix.get("speed_kmh", 18.0),
                    curr_fix.get("bearing_deg", 340.0),
                    curr_fix.get("dlat_6h", 0.6),
                    curr_fix.get("dlon_6h", -0.2),
                    curr_fix.get("dlat_12h", 1.1),
                    curr_fix.get("dlon_12h", -0.4),
                    curr_fix.get("dlat_24h", 2.2),
                    curr_fix.get("dlon_24h", -0.8),
                    curr_fix.get("dwind_6h", 5.0),
                    curr_fix.get("dwind_12h", 10.0),
                    curr_fix.get("dwind_24h", 15.0),
                    curr_fix.get("dpres_6h", -4.0),
                    curr_fix.get("dpres_12h", -8.0),
                    curr_fix.get("month", 10),
                    curr_fix.get("sin_doy", 0.8),
                    curr_fix.get("cos_doy", 0.5),
                ]
                X_sample = np.array(feat_vals, dtype=np.float32).reshape(1, -1)

                cone_radii = {6: 48.0, 12: 94.0, 24: 187.0, 48: 404.0, 72: 620.0}
                forecast_points = []
                c_lat = float(curr_fix.get("lat", 17.5))
                c_lon = float(curr_fix.get("lon", 88.5))
                c_wind = float(curr_fix.get("wind_kt", 60.0))
                c_pres = float(curr_fix.get("pres_hpa", 985.0))

                for h in [6, 12, 24, 48, 72]:
                    if h in cyclone_model.models_dlat:
                        p_dlat = float(cyclone_model.models_dlat[h].predict(X_sample)[0])
                        p_dlon = float(cyclone_model.models_dlon[h].predict(X_sample)[0])
                        p_dwind = float(cyclone_model.models_dwind[h].predict(X_sample)[0])
                        p_dpres = float(cyclone_model.models_dpres[h].predict(X_sample)[0])

                        f_lat = round(c_lat + p_dlat, 2)
                        f_lon = round(c_lon + p_dlon, 2)
                        f_wind = round(max(15.0, c_wind + p_dwind), 1)
                        f_pres = round(min(1020.0, c_pres + p_dpres), 1)
                        c_radius = cone_radii.get(h, 100.0)

                        forecast_points.append({
                            "horizon_h": h,
                            "lat": f_lat,
                            "lon": f_lon,
                            "wind_kt": f_wind,
                            "pres_hpa": f_pres,
                            "cone_radius_km": c_radius
                        })

                # Assess Coastal Exposure & Flood Inundation
                exposure_res = exposure_engine.assess_storm_impact(forecast_points)

                # Query local LLM (qwen2.5:0.5b GGUF via Ollama) for emergency disaster bulletin
                s = exposure_res["summary"]
                llm_bulletin = query_ollama_cyclone_bulletin(
                    storm_name=storm_name or "Tropical Storm",
                    max_wind=s.get("max_wind_kt", 60.0),
                    min_pres=s.get("min_pres_hpa", 985.0),
                    surge_m=s.get("projected_storm_surge_m", 1.5),
                    affected_dists=s.get("total_coastal_districts_affected", 0),
                    exposed_ha=s.get("total_exposed_cropland_ha", 0)
                )
                if llm_bulletin:
                    exposure_res["disaster_bulletin"] = llm_bulletin
                    exposure_res["bulletin_source"] = "Ollama LLM (qwen2.5:0.5b GGUF Neural Engine)"
                else:
                    exposure_res["disaster_bulletin"] = "Maintain high alert across coastal sectors. Expedite drainage of low-lying paddy and harvest standing mature crops immediately."
                    exposure_res["bulletin_source"] = "Coastal Emergency Advisory Protocol"

                response = {
                    "current_fix": {
                        "lat": c_lat,
                        "lon": c_lon,
                        "wind_kt": c_wind,
                        "pres_hpa": c_pres
                    },
                    "historical_track": hist_points[-15:],
                    "forecast_track": forecast_points,
                    "exposure": exposure_res
                }
                self.send_json_response(response)
            except Exception as e:
                logger.error(f"Error in /api/predict_cyclone: {e}", exc_info=True)
                self.send_json_response({"error": str(e)}, status=500)

        else:
            self.send_error(404, "Endpoint not found")

    def send_json_response(self, data: Any, status: int = 200):
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


def run_dashboard_server(port: int = 8000):
    """Starts the threaded HTTP server on the specified port."""
    initialize_engines()
    server_address = ("0.0.0.0", port)
    httpd = ThreadedHTTPServer(server_address, DashboardRequestHandler)
    print("=" * 80)
    print(f"INDRA DUAL PREDICTIVE PLATFORM SERVER RUNNING AT: http://localhost:{port}")
    print(f"Endpoints:")
    print(f"  • UI Dashboard:            http://localhost:{port}/")
    print(f"  • Crop Prediction API:    POST http://localhost:{port}/api/predict_crop")
    print(f"  • Cyclone Forecast API:   POST http://localhost:{port}/api/predict_cyclone")
    print(f"  • System Status API:      GET  http://localhost:{port}/api/status")
    print("=" * 80)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server...")
        httpd.server_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run_dashboard_server(port=args.port)
