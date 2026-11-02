"""Common scientific evaluation metrics for Crop Yield Prediction and Disaster Forecasting."""
from __future__ import annotations

import math
from typing import Dict, Tuple, Union
import numpy as np
import pandas as pd


# ==============================================================================
# GEOSPATIAL & CYCLONE METRICS
# ==============================================================================

def haversine_distance(
    lat1: Union[float, np.ndarray],
    lon1: Union[float, np.ndarray],
    lat2: Union[float, np.ndarray],
    lon2: Union[float, np.ndarray],
    radius_km: float = 6371.0088
) -> Union[float, np.ndarray]:
    """Calculate the great-circle distance between two points on the Earth (Haversine formula).

    Parameters:
        lat1, lon1: Coordinates of the first point(s) in decimal degrees.
        lat2, lon2: Coordinates of the second point(s) in decimal degrees.
        radius_km: Earth radius in kilometers (default: 6371.0088 km).

    Returns:
        Great-circle distance in kilometers.
    """
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * (np.sin(delta_lambda / 2.0) ** 2)
    )
    # Clip to avoid numerical precision errors exceeding 1.0
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return radius_km * c


def mean_track_error(
    pred_lat: np.ndarray,
    pred_lon: np.ndarray,
    true_lat: np.ndarray,
    true_lon: np.ndarray
) -> float:
    """Compute the Mean Track Error (MTE) in kilometers across forecast points."""
    distances = haversine_distance(pred_lat, pred_lon, true_lat, true_lon)
    return float(np.nanmean(distances))


def along_cross_track_errors(
    pred_lat: np.ndarray,
    pred_lon: np.ndarray,
    true_lat: np.ndarray,
    true_lon: np.ndarray,
    prev_true_lat: np.ndarray,
    prev_true_lon: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Decompose forecast displacement into along-track (speed) and cross-track (direction) errors.

    Positive along-track error means forecast is ahead of actual storm movement.
    Positive cross-track error indicates displacement to the right of the actual motion vector.
    """
    # Vector of actual motion
    d_lat_actual = true_lat - prev_true_lat
    d_lon_actual = (true_lon - prev_true_lon) * np.cos(np.radians(true_lat))
    track_angle = np.arctan2(d_lat_actual, d_lon_actual)

    # Forecast displacement relative to true location
    d_lat_err = pred_lat - true_lat
    d_lon_err = (pred_lon - true_lon) * np.cos(np.radians(true_lat))

    # Convert degrees to approximate km (~111.13 km per degree)
    KM_PER_DEG = 111.132954
    dy_km = d_lat_err * KM_PER_DEG
    dx_km = d_lon_err * KM_PER_DEG

    # Rotate coordinate system into along-track and cross-track
    along_track = dx_km * np.cos(track_angle) + dy_km * np.sin(track_angle)
    cross_track = -dx_km * np.sin(track_angle) + dy_km * np.cos(track_angle)

    return along_track, cross_track


def cyclone_intensity_metrics(
    pred_val: np.ndarray,
    true_val: np.ndarray
) -> Dict[str, float]:
    """Compute MAE, RMSE, and bias for intensity variables (wind speed in kt, central pressure in hPa)."""
    valid_mask = ~np.isnan(pred_val) & ~np.isnan(true_val)
    if not np.any(valid_mask):
        return {"mae": np.nan, "rmse": np.nan, "bias": np.nan, "count": 0}

    diff = pred_val[valid_mask] - true_val[valid_mask]
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    bias = float(np.mean(diff))

    return {"mae": mae, "rmse": rmse, "bias": bias, "count": int(np.sum(valid_mask))}


# ==============================================================================
# CROP YIELD & REGRESSION METRICS
# ==============================================================================

def crop_yield_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Compute standard agricultural yield forecasting evaluation metrics (kg/ha).

    Metrics:
        - MAE: Mean Absolute Error (kg/ha)
        - RMSE: Root Mean Squared Error (kg/ha)
        - R2: Coefficient of Determination
        - WAPE: Weighted Absolute Percentage Error (%)
        - MAPE: Mean Absolute Percentage Error (%)
        - Pearson_r: Pearson correlation coefficient
    """
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & (y_true > 0)
    yt = y_true[valid_mask]
    yp = y_pred[valid_mask]

    if len(yt) < 2:
        return {"mae": np.nan, "rmse": np.nan, "r2": np.nan, "wape": np.nan, "mape": np.nan}

    diff = yt - yp
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    wape = float(np.sum(np.abs(diff)) / np.sum(yt)) * 100.0
    mape = float(np.mean(np.abs(diff) / yt)) * 100.0

    # R2
    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    # Pearson r
    std_t = np.std(yt)
    std_p = np.std(yp)
    if std_t > 0 and std_p > 0:
        pearson_r = float(np.corrcoef(yt, yp)[0, 1])
    else:
        pearson_r = 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "wape_pct": wape,
        "mape_pct": mape,
        "pearson_r": pearson_r,
        "sample_count": len(yt)
    }


def prediction_interval_coverage(
    y_true: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray
) -> Dict[str, float]:
    """Compute Prediction Interval Coverage Probability (PICP) and Mean Prediction Interval Width (MPIW)."""
    valid = ~np.isnan(y_true) & ~np.isnan(lower_bound) & ~np.isnan(upper_bound)
    yt = y_true[valid]
    lb = lower_bound[valid]
    ub = upper_bound[valid]

    covered = (yt >= lb) & (yt <= ub)
    picp = float(np.mean(covered)) * 100.0
    mpiw = float(np.mean(ub - lb))

    return {"picp_pct": picp, "mpiw": mpiw, "sample_count": len(yt)}
