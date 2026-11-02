"""Data cleaning and disaggregation pipeline for district-level agricultural data (UPAg).

Filters out macro-commodity aggregates (Total Food Grains, Cereals, etc.) and
aggregate seasons (Total), standardizes district names and units, validates derived yield,
and exports clean crop-specific panels.
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

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Macro aggregates to separate from true biological crops
MACRO_COMMODITY_GROUPS = [
    "Total Food Grains",
    "Cereals",
    "Nutri/Coarse Cereals",
    "Total Pulses",
    "Shree Anna /Nutri Cereals",
    "Other Pulses",
]

TRUE_CROPS = [
    "Rice",
    "Wheat",
    "Maize",
    "Gram",
    "Tur",
    "Urad",
    "Moong",
    "Jowar",
    "Bajra",
    "Ragi",
    "Lentil",
    "Barley",
    "Small Millets",
]


class UPAgDistrictCleaner:
    """Preprocesses UPAg 2024-25 district data and generates clean analytical tables."""

    def __init__(
        self,
        raw_csv_path: Union[str, Path] = "/home/neymar-jr/DSC Project/UPAg/DES-District-Data-For-2024-25.csv",
        output_dir: Union[str, Path] = "/home/neymar-jr/DSC Project/data_processed"
    ):
        self.raw_csv_path = Path(raw_csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads, cleans, and splits district data into crop-specific panel and commodity aggregates."""
        if not self.raw_csv_path.exists():
            raise FileNotFoundError(f"UPAg file not found at: {self.raw_csv_path}")

        df = pd.read_csv(self.raw_csv_path, low_memory=False)

        # Standardize column names
        rename_map = {
            "State": "state",
            "District": "district",
            "Crop": "crop",
            "Season": "season",
            "Area-2024-25": "area_thousand_ha",
            "Production-2024-25": "production_thousand_tonnes",
            "Yield-2024-25": "yield_kg_per_ha",
        }
        df = df.rename(columns=rename_map)

        # Clean string columns
        for c in ["state", "district", "crop", "season"]:
            df[c] = df[c].astype(str).str.strip().str.title()

        # Parse numerics
        num_cols = ["area_thousand_ha", "production_thousand_tonnes", "yield_kg_per_ha"]
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Validate and derive yield:
        # Area is in Thousand Hectares (1000 ha)
        # Production is in Thousand Tonnes (1000 t)
        # Derived Yield (kg/ha) = (Production * 1000 * 1000 kg) / (Area * 1000 ha) = (Production / Area) * 1000
        valid_area = df["area_thousand_ha"] > 0
        valid_prod = df["production_thousand_tonnes"] > 0
        df["derived_yield_kg_per_ha"] = np.nan
        df.loc[valid_area & valid_prod, "derived_yield_kg_per_ha"] = (
            (df.loc[valid_area & valid_prod, "production_thousand_tonnes"] /
             df.loc[valid_area & valid_prod, "area_thousand_ha"]) * 1000.0
        )

        # Impute missing reported yield with derived yield when available
        df["yield_kg_per_ha"] = df["yield_kg_per_ha"].combine_first(df["derived_yield_kg_per_ha"])

        # Split: Macro Aggregates vs True Crops
        is_macro = df["crop"].isin([c.title() for c in MACRO_COMMODITY_GROUPS])
        is_total_season = df["season"] == "Total"

        # 1. Macro commodity summaries
        macro_df = df[is_macro | is_total_season].copy().reset_index(drop=True)

        # 2. Individual crop-specific analytical dataset
        crop_df = df[(~is_macro) & (~is_total_season) & (df["area_thousand_ha"] > 0)].copy()
        crop_df = crop_df.dropna(subset=["yield_kg_per_ha"]).reset_index(drop=True)

        # Add year column
        crop_df["year"] = "2024-25"
        macro_df["year"] = "2024-25"

        # Save clean datasets
        crop_parquet = self.output_dir / "upag_district_cleaned_2024_25.parquet"
        crop_csv = self.output_dir / "upag_district_cleaned_2024_25.csv"
        macro_parquet = self.output_dir / "upag_macro_aggregates_2024_25.parquet"

        crop_df.to_parquet(crop_parquet, index=False)
        crop_df.to_csv(crop_csv, index=False)
        macro_df.to_parquet(macro_parquet, index=False)

        logger.info(f"Saved {len(crop_df)} clean crop-district records to {crop_parquet}")
        logger.info(f"Saved {len(macro_df)} macro-aggregate records to {macro_parquet}")

        return crop_df, macro_df


if __name__ == "__main__":
    cleaner = UPAgDistrictCleaner()
    crop_df, macro_df = cleaner.process()
    print("=" * 80)
    print("UPAg DISTRICT DATA CLEANING COMPLETE")
    print("=" * 80)
    print(f"Clean Crop Records: {len(crop_df)} rows")
    print(f"Crops: {crop_df['crop'].nunique()} distinct crops ({sorted(crop_df['crop'].unique())})")
    print(f"States: {crop_df['state'].nunique()} | Districts: {crop_df['district'].nunique()}")
    print(f"Seasons: {crop_df['season'].value_counts().to_dict()}")
    print("\nYield Summary (kg/ha):")
    print(crop_df["yield_kg_per_ha"].describe())
