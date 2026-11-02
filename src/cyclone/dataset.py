"""Dataset builder and feature engineering pipeline for North Indian Ocean Tropical Cyclones.

Processes IBTrACS observations, handles agency harmonization (IMD vs JTWC), regularizes tracks,
engineers kinematic lag features, and constructs multi-horizon forecasting targets.
"""
from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Ensure DSC Project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.common.metrics import haversine_distance

logger = logging.getLogger(__name__)


def calculate_bearing(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray
) -> np.ndarray:
    """Calculate the forward azimuth / compass bearing in degrees between two points."""
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_lambda = np.radians(lon2 - lon1)

    y = np.sin(delta_lambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_lambda)
    bearing_rad = np.arctan2(y, x)
    bearing_deg = (np.degrees(bearing_rad) + 360.0) % 360.0
    return bearing_deg


class CycloneDatasetBuilder:
    """Ingests IBTrACS North Indian Ocean storm tracks and creates multi-horizon datasets."""

    def __init__(
        self,
        raw_csv_path: Union[str, Path] = "/home/neymar-jr/DSC Project/dataset/ibtracs.NI.list.v04r01.csv",
        start_year: int = 1990,
        horizons_hours: List[int] = [6, 12, 24, 48, 72]
    ):
        self.raw_csv_path = Path(raw_csv_path)
        self.start_year = start_year
        self.horizons_hours = sorted(horizons_hours)

    def load_clean_data(self) -> pd.DataFrame:
        """Load and clean IBTrACS North Indian Ocean observations."""
        if not self.raw_csv_path.exists():
            raise FileNotFoundError(f"IBTrACS file not found at: {self.raw_csv_path}")

        # Line 1 contains column headers; Line 2 contains units (skiprows=[1])
        df = pd.read_csv(self.raw_csv_path, skiprows=[1], low_memory=False)

        # Parse year and filter to modern satellite era
        df["SEASON_NUM"] = pd.to_numeric(df["SEASON"], errors="coerce")
        df = df[df["SEASON_NUM"] >= self.start_year].copy()

        # Parse timestamp
        df["datetime"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
        df = df.dropna(subset=["datetime", "LAT", "LON", "SID"]).sort_values(["SID", "datetime"])

        # Numeric coordinates
        df["lat"] = pd.to_numeric(df["LAT"], errors="coerce")
        df["lon"] = pd.to_numeric(df["LON"], errors="coerce")

        # Wind speed harmonization (kt): IMD (WMO_WIND) prioritized, fallback to JTWC (USA_WIND)
        wind_wmo = pd.to_numeric(df["WMO_WIND"], errors="coerce")
        wind_usa = pd.to_numeric(df["USA_WIND"], errors="coerce")
        df["wind_kt"] = wind_wmo.combine_first(wind_usa)

        # Pressure harmonization (hPa): IMD (WMO_PRES) prioritized, fallback to JTWC (USA_PRES)
        pres_wmo = pd.to_numeric(df["WMO_PRES"], errors="coerce")
        pres_usa = pd.to_numeric(df["USA_PRES"], errors="coerce")
        df["pres_hpa"] = pres_wmo.combine_first(pres_usa)

        # Distance to land (km)
        df["dist2land_km"] = pd.to_numeric(df["DIST2LAND"], errors="coerce").fillna(0)
        df["landfall"] = pd.to_numeric(df["LANDFALL"], errors="coerce").fillna(0)

        # Interpolate missing wind and pressure per storm linearly where gaps are short
        df["wind_kt"] = df.groupby("SID")["wind_kt"].transform(
            lambda s: s.interpolate(method="linear", limit=4).bfill().ffill()
        )
        df["pres_hpa"] = df.groupby("SID")["pres_hpa"].transform(
            lambda s: s.interpolate(method="linear", limit=4).bfill().ffill()
        )

        # Remove any remaining missing values in essential coordinates or wind
        df = df.dropna(subset=["lat", "lon", "wind_kt"]).reset_index(drop=True)
        return df

    def extract_features_and_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer lag kinematics, environmental proxies, and multi-horizon prediction targets."""
        processed_rows = []

        for sid, storm in df.groupby("SID"):
            storm = storm.sort_values("datetime").reset_index(drop=True)
            n_obs = len(storm)
            if n_obs < 4:
                # Need at least 4 observations (~12-24h) for a meaningful trajectory sequence
                continue

            # Compute observation intervals
            dt_hours = (storm["datetime"].diff().dt.total_seconds() / 3600.0).fillna(6.0).values

            lats = storm["lat"].values
            lons = storm["lon"].values
            winds = storm["wind_kt"].values
            press = storm["pres_hpa"].fillna(1005.0).values
            d2land = storm["dist2land_km"].values
            times = storm["datetime"].values
            times_sec = storm["datetime"].values.astype("datetime64[s]").astype("int64")
            names = storm["NAME"].values
            years = storm["SEASON_NUM"].values

            # Step-by-step feature generation
            for i in range(1, n_obs):
                t_curr = times[i]
                t_curr_sec = times_sec[i]
                lat_curr = lats[i]
                lon_curr = lons[i]
                wind_curr = winds[i]
                pres_curr = press[i]
                d2l_curr = d2land[i]

                # 1. Past displacements
                lat_prev = lats[i - 1]
                lon_prev = lons[i - 1]
                dlat_6h = lat_curr - lat_prev
                dlon_6h = lon_curr - lon_prev
                dist_prev_km = haversine_distance(lat_prev, lon_prev, lat_curr, lon_curr)
                dt_prev = max(1.0, dt_hours[i])
                speed_kmh = float(dist_prev_km / dt_prev)
                bearing = float(calculate_bearing(np.array([lat_prev]), np.array([lon_prev]), np.array([lat_curr]), np.array([lon_curr]))[0])

                if i >= 2:
                    dlat_12h = lat_curr - lats[i - 2]
                    dlon_12h = lon_curr - lons[i - 2]
                    dwind_12h = wind_curr - winds[i - 2]
                    dpres_12h = pres_curr - press[i - 2]
                else:
                    dlat_12h = dlat_6h * 2.0
                    dlon_12h = dlon_6h * 2.0
                    dwind_12h = (wind_curr - winds[i - 1]) * 2.0
                    dpres_12h = (pres_curr - press[i - 1]) * 2.0

                if i >= 4:
                    dlat_24h = lat_curr - lats[i - 4]
                    dlon_24h = lon_curr - lons[i - 4]
                    dwind_24h = wind_curr - winds[i - 4]
                else:
                    dlat_24h = dlat_12h * 2.0
                    dlon_24h = dlon_12h * 2.0
                    dwind_24h = dwind_12h * 2.0

                dwind_6h = wind_curr - winds[i - 1]
                dpres_6h = pres_curr - press[i - 1]

                ts = pd.Timestamp(t_curr)
                month = ts.month
                day_of_year = ts.day_of_year
                sin_doy = math.sin(2.0 * math.pi * day_of_year / 365.25)
                cos_doy = math.cos(2.0 * math.pi * day_of_year / 365.25)

                row_dict = {
                    "sid": sid,
                    "name": names[i],
                    "year": years[i],
                    "datetime": t_curr,
                    "lat": lat_curr,
                    "lon": lon_curr,
                    "wind_kt": wind_curr,
                    "pres_hpa": pres_curr,
                    "dist2land_km": d2l_curr,
                    "speed_kmh": speed_kmh,
                    "bearing_deg": bearing,
                    "dlat_6h": dlat_6h,
                    "dlon_6h": dlon_6h,
                    "dlat_12h": dlat_12h,
                    "dlon_12h": dlon_12h,
                    "dlat_24h": dlat_24h,
                    "dlon_24h": dlon_24h,
                    "dwind_6h": dwind_6h,
                    "dwind_12h": dwind_12h,
                    "dwind_24h": dwind_24h,
                    "dpres_6h": dpres_6h,
                    "dpres_12h": dpres_12h,
                    "month": month,
                    "sin_doy": sin_doy,
                    "cos_doy": cos_doy,
                }

                # Multi-horizon forward targets using O(log N) searchsorted
                for h in self.horizons_hours:
                    target_sec = t_curr_sec + h * 3600
                    idx = np.searchsorted(times_sec, target_sec)
                    best_match = None
                    min_diff = 3 * 3600  # within +/- 3 hours tolerance

                    for cand in [idx - 1, idx, idx + 1]:
                        if 0 <= cand < n_obs:
                            diff = abs(times_sec[cand] - target_sec)
                            if diff <= min_diff:
                                min_diff = diff
                                best_match = cand

                    if best_match is not None and best_match > i:
                        fut_lat = lats[best_match]
                        fut_lon = lons[best_match]
                        fut_wind = winds[best_match]
                        fut_pres = press[best_match]

                        row_dict[f"target_lat_{h}h"] = fut_lat
                        row_dict[f"target_lon_{h}h"] = fut_lon
                        row_dict[f"target_dlat_{h}h"] = fut_lat - lat_curr
                        row_dict[f"target_dlon_{h}h"] = fut_lon - lon_curr
                        row_dict[f"target_wind_{h}h"] = fut_wind
                        row_dict[f"target_dwind_{h}h"] = fut_wind - wind_curr
                        row_dict[f"target_pres_{h}h"] = fut_pres
                        row_dict[f"target_valid_{h}h"] = 1
                    else:
                        row_dict[f"target_lat_{h}h"] = np.nan
                        row_dict[f"target_lon_{h}h"] = np.nan
                        row_dict[f"target_dlat_{h}h"] = np.nan
                        row_dict[f"target_dlon_{h}h"] = np.nan
                        row_dict[f"target_wind_{h}h"] = np.nan
                        row_dict[f"target_dwind_{h}h"] = np.nan
                        row_dict[f"target_pres_{h}h"] = np.nan
                        row_dict[f"target_valid_{h}h"] = 0

                processed_rows.append(row_dict)

        result_df = pd.DataFrame(processed_rows)
        return result_df

    def split_by_event(
        self,
        df: pd.DataFrame,
        train_end_year: int = 2020,
        val_end_year: int = 2022
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split into train, validation, and test partitions by calendar year / storm event.

        Strictly prevents temporal and track leakage:
        - Train: Storms <= train_end_year (e.g. 1990-2020)
        - Validation: train_end_year < Storms <= val_end_year (e.g. 2021-2022)
        - Test: Storms > val_end_year (e.g. 2023-2024+, including Mocha, Biparjoy, Remal, Dana)
        """
        train = df[df["year"] <= train_end_year].copy().reset_index(drop=True)
        val = df[(df["year"] > train_end_year) & (df["year"] <= val_end_year)].copy().reset_index(drop=True)
        test = df[df["year"] > val_end_year].copy().reset_index(drop=True)

        return train, val, test


if __name__ == "__main__":
    builder = CycloneDatasetBuilder()
    clean_df = builder.load_clean_data()
    print(f"Loaded {len(clean_df)} clean points from {clean_df['SID'].nunique()} storms.")
    feature_df = builder.extract_features_and_targets(clean_df)
    print(f"Engineered features for {len(feature_df)} trajectory points.")
    train, val, test = builder.split_by_event(feature_df)
    print(f"Split counts: Train={len(train)} (years <=2020), Val={len(val)} (2021-2022), Test={len(test)} (2023+)")
