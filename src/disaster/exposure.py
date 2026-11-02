"""Coastal Multi-Hazard Disaster Exposure & Agricultural Impact Assessment Engine.

Models:
1. Cyclone wind field radius (Gale 34kt, Storm 48kt, Hurricane 64kt).
2. Storm surge height potential (SLOSH-formulated empirical pressure/wind dynamic).
3. Geographical intersection with Indian coastal district agricultural assets (UPAg).
4. Estimation of potentially inundated agricultural cropland (hectares) and affected crops.
5. Calibrated hazard severity tiers and automated disaster relief priority scoring.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.common.metrics import haversine_distance
from src.disaster.dem_processor import SRTMProcessor

logger = logging.getLogger(__name__)

# Master Coastal Districts Geo-Database with Agricultural Profiles (derived from UPAg records)
COASTAL_DISTRICTS = [
    # West Bengal
    {"state": "West Bengal", "district": "South 24 Parganas", "lat": 22.15, "lon": 88.55, "total_cropland_ha": 372000, "dominant_crops": ["Rice", "Jute", "Vegetables"], "pop_density_per_sqkm": 820},
    {"state": "West Bengal", "district": "Purba Medinipur", "lat": 21.95, "lon": 87.75, "total_cropland_ha": 298000, "dominant_crops": ["Rice", "Betelvine", "Groundnut"], "pop_density_per_sqkm": 1080},
    {"state": "West Bengal", "district": "North 24 Parganas", "lat": 22.72, "lon": 88.48, "total_cropland_ha": 265000, "dominant_crops": ["Rice", "Jute", "Pulses"], "pop_density_per_sqkm": 2400},

    # Odisha
    {"state": "Odisha", "district": "Jagatsinghpur", "lat": 20.25, "lon": 86.17, "total_cropland_ha": 105000, "dominant_crops": ["Rice", "Pulses", "Sugarcane"], "pop_density_per_sqkm": 680},
    {"state": "Odisha", "district": "Kendrapara", "lat": 20.50, "lon": 86.42, "total_cropland_ha": 152000, "dominant_crops": ["Rice", "Jute", "Groundnut"], "pop_density_per_sqkm": 540},
    {"state": "Odisha", "district": "Bhadrak", "lat": 21.05, "lon": 86.50, "total_cropland_ha": 172000, "dominant_crops": ["Rice", "Mustard", "Vegetables"], "pop_density_per_sqkm": 600},
    {"state": "Odisha", "district": "Baleshwar", "lat": 21.50, "lon": 86.92, "total_cropland_ha": 240000, "dominant_crops": ["Rice", "Pulses", "Oilseeds"], "pop_density_per_sqkm": 610},
    {"state": "Odisha", "district": "Puri", "lat": 19.81, "lon": 85.83, "total_cropland_ha": 138000, "dominant_crops": ["Rice", "Coconut", "Groundnut"], "pop_density_per_sqkm": 490},
    {"state": "Odisha", "district": "Ganjam", "lat": 19.38, "lon": 85.05, "total_cropland_ha": 395000, "dominant_crops": ["Rice", "Ragi", "Pulses"], "pop_density_per_sqkm": 430},

    # Andhra Pradesh
    {"state": "Andhra Pradesh", "district": "Srikakulam", "lat": 18.30, "lon": 83.90, "total_cropland_ha": 320000, "dominant_crops": ["Rice", "Groundnut", "Sugarcane"], "pop_density_per_sqkm": 460},
    {"state": "Andhra Pradesh", "district": "Visakhapatnam", "lat": 17.69, "lon": 83.22, "total_cropland_ha": 185000, "dominant_crops": ["Rice", "Sugarcane", "Millets"], "pop_density_per_sqkm": 380},
    {"state": "Andhra Pradesh", "district": "East Godavari", "lat": 16.98, "lon": 82.24, "total_cropland_ha": 410000, "dominant_crops": ["Rice", "Coconut", "Cotton"], "pop_density_per_sqkm": 510},
    {"state": "Andhra Pradesh", "district": "West Godavari", "lat": 16.71, "lon": 81.10, "total_cropland_ha": 380000, "dominant_crops": ["Rice", "Sugarcane", "Maize"], "pop_density_per_sqkm": 500},
    {"state": "Andhra Pradesh", "district": "Krishna", "lat": 16.18, "lon": 81.13, "total_cropland_ha": 445000, "dominant_crops": ["Rice", "Cotton", "Pulses"], "pop_density_per_sqkm": 520},
    {"state": "Andhra Pradesh", "district": "Guntur", "lat": 16.30, "lon": 80.44, "total_cropland_ha": 490000, "dominant_crops": ["Cotton", "Chilli", "Rice"], "pop_density_per_sqkm": 430},
    {"state": "Andhra Pradesh", "district": "Prakasam", "lat": 15.50, "lon": 80.05, "total_cropland_ha": 520000, "dominant_crops": ["Tobacco", "Bengal Gram", "Cotton"], "pop_density_per_sqkm": 190},
    {"state": "Andhra Pradesh", "district": "Sps Nellore", "lat": 14.44, "lon": 79.98, "total_cropland_ha": 340000, "dominant_crops": ["Rice", "Groundnut", "Sugarcane"], "pop_density_per_sqkm": 230},

    # Tamil Nadu
    {"state": "Tamil Nadu", "district": "Tiruvallur", "lat": 13.14, "lon": 79.91, "total_cropland_ha": 145000, "dominant_crops": ["Rice", "Groundnut", "Pulses"], "pop_density_per_sqkm": 1100},
    {"state": "Tamil Nadu", "district": "Cuddalore", "lat": 11.75, "lon": 79.77, "total_cropland_ha": 235000, "dominant_crops": ["Rice", "Sugarcane", "Cashew"], "pop_density_per_sqkm": 700},
    {"state": "Tamil Nadu", "district": "Nagapattinam", "lat": 10.77, "lon": 79.84, "total_cropland_ha": 165000, "dominant_crops": ["Rice", "Pulses", "Groundnut"], "pop_density_per_sqkm": 600},
    {"state": "Tamil Nadu", "district": "Thanjavur", "lat": 10.79, "lon": 79.14, "total_cropland_ha": 280000, "dominant_crops": ["Rice", "Coconut", "Groundnut"], "pop_density_per_sqkm": 710},
    {"state": "Tamil Nadu", "district": "Ramanathapuram", "lat": 9.37, "lon": 78.84, "total_cropland_ha": 178000, "dominant_crops": ["Rice", "Cotton", "Chilli"], "pop_density_per_sqkm": 330},

    # Gujarat
    {"state": "Gujarat", "district": "Kachchh", "lat": 23.25, "lon": 69.67, "total_cropland_ha": 650000, "dominant_crops": ["Castor", "Cotton", "Groundnut"], "pop_density_per_sqkm": 46},
    {"state": "Gujarat", "district": "Devbhumi Dwarka", "lat": 22.24, "lon": 68.97, "total_cropland_ha": 260000, "dominant_crops": ["Groundnut", "Cotton", "Wheat"], "pop_density_per_sqkm": 180},
    {"state": "Gujarat", "district": "Jamnagar", "lat": 22.47, "lon": 70.07, "total_cropland_ha": 310000, "dominant_crops": ["Cotton", "Groundnut", "Sesamum"], "pop_density_per_sqkm": 210},
    {"state": "Gujarat", "district": "Porbandar", "lat": 21.64, "lon": 69.61, "total_cropland_ha": 135000, "dominant_crops": ["Groundnut", "Cotton", "Wheat"], "pop_density_per_sqkm": 260},
    {"state": "Gujarat", "district": "Junagadh", "lat": 21.52, "lon": 70.46, "total_cropland_ha": 330000, "dominant_crops": ["Groundnut", "Cotton", "Wheat"], "pop_density_per_sqkm": 310},
    {"state": "Gujarat", "district": "Gir Somnath", "lat": 20.90, "lon": 70.37, "total_cropland_ha": 190000, "dominant_crops": ["Groundnut", "Cotton", "Sugarcane"], "pop_density_per_sqkm": 320},
    {"state": "Gujarat", "district": "Bhavnagar", "lat": 21.76, "lon": 72.15, "total_cropland_ha": 420000, "dominant_crops": ["Cotton", "Groundnut", "Onion"], "pop_density_per_sqkm": 290},
    {"state": "Gujarat", "district": "Surat", "lat": 21.17, "lon": 72.83, "total_cropland_ha": 210000, "dominant_crops": ["Sugarcane", "Rice", "Banana"], "pop_density_per_sqkm": 1380},
    {"state": "Gujarat", "district": "Valsad", "lat": 20.61, "lon": 72.93, "total_cropland_ha": 140000, "dominant_crops": ["Rice", "Mango", "Sugarcane"], "pop_density_per_sqkm": 560},
]


class CoastalExposureEngine:
    """Computes coastal multi-hazard impacts, surge heights, and agricultural land exposure."""

    def __init__(self, srtm_processor: Optional[SRTMProcessor] = None):
        self.srtm = srtm_processor or SRTMProcessor()
        self.coastal_df = pd.DataFrame(COASTAL_DISTRICTS)

    def calculate_surge_height(self, max_wind_kt: float, min_pres_hpa: float) -> float:
        """SLOSH / Jelesnianski empirical dynamic formulation for tropical cyclone storm surge.
        
        S_surge = 0.01 * (1013 - P_min) + 0.005 * (V_max^1.5)
        """
        pres_component = 0.01 * max(0.0, 1013.25 - min_pres_hpa)
        wind_component = 0.005 * (max(0.0, max_wind_kt) ** 1.5)
        surge_m = max(0.5, pres_component + wind_component)
        return round(float(surge_m), 2)

    def compute_wind_radii(self, max_wind_kt: float) -> Dict[str, float]:
        """Estimates radial extent of gale (34kt), storm (48kt), and hurricane (64kt) force winds in km."""
        if max_wind_kt < 34.0:
            return {"gale_radius_km": 0.0, "storm_radius_km": 0.0, "hurricane_radius_km": 0.0}

        # Empirical wind-pressure radial scaling (Willoughby / Holland formulation)
        scale = np.sqrt(max_wind_kt / 34.0)
        r34 = min(450.0, 160.0 * scale)
        r48 = min(250.0, 85.0 * scale) if max_wind_kt >= 48.0 else 0.0
        r64 = min(120.0, 45.0 * scale) if max_wind_kt >= 64.0 else 0.0

        return {
            "gale_radius_km": round(float(r34), 1),
            "storm_radius_km": round(float(r48), 1),
            "hurricane_radius_km": round(float(r64), 1)
        }

    def assess_storm_impact(
        self,
        forecast_points: List[Dict[str, float]],
        surge_height_m: Optional[float] = None
    ) -> Dict[str, Any]:
        """Calculates district-level agricultural cropland exposure and hazard priority."""
        if not forecast_points:
            return {"exposed_districts": [], "total_exposed_cropland_ha": 0}

        # Track trajectory bounding properties
        all_lats = [p["lat"] for p in forecast_points]
        all_lons = [p["lon"] for p in forecast_points]
        max_wind = max(p.get("wind_kt", 45.0) for p in forecast_points)
        min_pres = min(p.get("pres_hpa", 990.0) for p in forecast_points)

        radii = self.compute_wind_radii(max_wind)
        r34 = radii["gale_radius_km"]
        r48 = radii["storm_radius_km"]
        r64 = radii["hurricane_radius_km"]

        calculated_surge = surge_height_m or self.calculate_surge_height(max_wind, min_pres)

        exposed_districts = []
        total_exposed_cropland = 0.0

        # Evaluate each coastal district against trajectory path
        for _, row in self.coastal_df.iterrows():
            d_lat = row["lat"]
            d_lon = row["lon"]

            # Compute minimum distance to any point along the forecast trajectory
            min_dist = min(
                haversine_distance(d_lat, d_lon, p["lat"], p["lon"])
                for p in forecast_points
            )

            # Determine hazard impact band
            threat_level = "None"
            exposure_fraction = 0.0

            if min_dist <= r64:
                threat_level = "Severe Hurricane Force (>64 kt)"
                exposure_fraction = 0.95
            elif min_dist <= r48:
                threat_level = "Destructive Storm Force (48-64 kt)"
                exposure_fraction = 0.65
            elif min_dist <= r34:
                threat_level = "Gale Force Wind (34-48 kt)"
                exposure_fraction = 0.30
            elif min_dist <= r34 + 60.0:
                threat_level = "Peripheral Advisory (25-34 kt)"
                exposure_fraction = 0.10

            if exposure_fraction > 0.0:
                # Topographic vulnerability from SRTM DEM
                dem_prof = self.srtm.analyze_coastal_profile(d_lat, d_lon)
                elev_m = dem_prof["elevation_m"]

                # Surge inundation multiplier
                surge_inundation_risk = "Low"
                if calculated_surge >= elev_m:
                    surge_inundation_risk = "Direct Surge Inundation"
                    exposure_fraction = min(1.0, exposure_fraction * 1.3)
                elif calculated_surge >= (elev_m * 0.6):
                    surge_inundation_risk = "Moderate Tidal Inundation"
                    exposure_fraction = min(1.0, exposure_fraction * 1.15)

                exposed_ha = round(row["total_cropland_ha"] * exposure_fraction, 0)
                total_exposed_cropland += exposed_ha

                # Disaster Priority Score (0 - 100)
                priority_score = min(100.0, (
                    (exposure_fraction * 50.0) +
                    (min(1.0, calculated_surge / max(elev_m, 1.0)) * 30.0) +
                    (min(1.0, max_wind / 120.0) * 20.0)
                ))

                exposed_districts.append({
                    "state": row["state"],
                    "district": row["district"],
                    "lat": d_lat,
                    "lon": d_lon,
                    "closest_track_distance_km": round(min_dist, 1),
                    "threat_level": threat_level,
                    "elevation_m": elev_m,
                    "inundation_tier": dem_prof["inundation_tier"],
                    "surge_risk": surge_inundation_risk,
                    "total_cropland_ha": row["total_cropland_ha"],
                    "exposed_cropland_ha": exposed_ha,
                    "exposure_pct": round(exposure_fraction * 100.0, 1),
                    "dominant_crops": row["dominant_crops"],
                    "priority_score": round(priority_score, 1)
                })

        # Sort districts by priority score descending
        exposed_districts.sort(key=lambda x: x["priority_score"], reverse=True)

        return {
            "summary": {
                "max_wind_kt": max_wind,
                "min_pres_hpa": min_pres,
                "projected_storm_surge_m": calculated_surge,
                "wind_radii": radii,
                "total_coastal_districts_affected": len(exposed_districts),
                "total_exposed_cropland_ha": round(total_exposed_cropland, 0)
            },
            "exposed_districts": exposed_districts
        }


if __name__ == "__main__":
    engine = CoastalExposureEngine()

    # Test trajectory matching Cyclone Remal (May 2024 landfall near WB / Sundarbans)
    sample_trajectory = [
        {"lat": 18.5, "lon": 89.2, "wind_kt": 60.0, "pres_hpa": 988.0, "horizon_h": 6},
        {"lat": 19.8, "lon": 89.1, "wind_kt": 70.0, "pres_hpa": 980.0, "horizon_h": 12},
        {"lat": 21.2, "lon": 89.0, "wind_kt": 75.0, "pres_hpa": 974.0, "horizon_h": 24},
        {"lat": 22.3, "lon": 89.1, "wind_kt": 65.0, "pres_hpa": 982.0, "horizon_h": 48},
    ]

    res = engine.assess_storm_impact(sample_trajectory)
    s = res["summary"]
    print("COASTAL MULTI-HAZARD DISASTER IMPACT TEST:")
    print("=" * 75)
    print(f"Peak Wind: {s['max_wind_kt']} kt | Central Pres: {s['min_pres_hpa']} hPa | Surge Height: {s['projected_storm_surge_m']} m")
    print(f"Gale Radius (34kt): {s['wind_radii']['gale_radius_km']} km | Storm Radius (48kt): {s['wind_radii']['storm_radius_km']} km")
    print(f"Total Cropland Exposed: {s['total_exposed_cropland_ha']:,} hectares across {s['total_coastal_districts_affected']} districts")
    print("-" * 75)
    print(f"{'District':<20} | {'State':<14} | {'Threat Level':<24} | {'Exposed (ha)':<12} | {'Priority'}")
    print("-" * 75)
    for d in res["exposed_districts"][:8]:
        print(f"{d['district']:<20} | {d['state']:<14} | {d['threat_level']:<24} | {d['exposed_cropland_ha']:<12,g} | {d['priority_score']}")
    print("=" * 75)
