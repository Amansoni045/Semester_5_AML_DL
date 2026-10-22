from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("preprocessing")

# Column name mappings
RAW_COLUMNS_ORDERED: List[str] = [
    "Country",
    "Area",
    "Waterbody Type",
    "Date",
    "Ammonia (mg/l)",
    "Biochemical Oxygen Demand (mg/l)",
    "Dissolved Oxygen (mg/l)",
    "Orthophosphate (mg/l)",
    "pH (ph units)",
    "Temperature (cel)",
    "Nitrogen (mg/l)",
    "Nitrate (mg/l)",
    "CCME_Values",
    "CCME_WQI",
]

COLUMN_NAME_MAP: Dict[str, str] = {
    "Country": "country",
    "Area": "area",
    "Waterbody Type": "waterbody_type",
    "Date": "date",
    "Ammonia (mg/l)": "ammonia_mg_l",
    "Biochemical Oxygen Demand (mg/l)": "bod_mg_l",
    "Dissolved Oxygen (mg/l)": "do_mg_l",
    "Orthophosphate (mg/l)": "orthophosphate_mg_l",
    "pH (ph units)": "ph",
    "Temperature (cel)": "temperature_c",
    "Nitrogen (mg/l)": "nitrogen_mg_l",
    "Nitrate (mg/l)": "nitrate_mg_l",
    "CCME_Values": "ccme_values",
    "CCME_WQI": "ccme_wqi",
}

COLUMN_NAME_MAP_REVERSE: Dict[str, str] = {v: k for k, v in COLUMN_NAME_MAP.items()}

# Telemetry error sentinels found in empirical data audit
SENTINEL_VALUES: List[float] = [-999999.0, -9999.0, -999.0, -596.0, -99.9, -80.0]
SENTINEL_COLUMNS: List[str] = ["ph", "temperature_c", "do_mg_l"]

# Plausible domain bounds used for quality flagging
SUSPICIOUS_VALUE_RULES: Dict[str, Dict[str, float]] = {
    "ph": {"hard_min": 0.0, "hard_max": 14.0},
    "temperature_c": {"hard_min": -5.0, "hard_max": 45.0},
    "ammonia_mg_l": {"hard_min": 0.0, "hard_max": 500.0},
    "bod_mg_l": {"hard_min": 0.0, "hard_max": 1000.0},
    "do_mg_l": {"hard_min": 0.0, "hard_max": 30.0},
    "orthophosphate_mg_l": {"hard_min": 0.0, "hard_max": 200.0},
    "nitrogen_mg_l": {"hard_min": 0.0, "hard_max": 500.0},
    "nitrate_mg_l": {"hard_min": 0.0, "hard_max": 1000.0},
    "ccme_values": {"hard_min": 0.0, "hard_max": 100.0},
}

# Column-specific cleaning rules for negative values
# Concentrations cannot be negative. Values in [noise_min, 0.0) reflect instrument zero-drift
# around the analytical detection limit and are clipped to 0.0. Values below noise_min are
# confirmed physically invalid (or hardware sentinels) and converted to NaN.
# Temperature is excluded here because sub-zero water/slush temperatures down to -10C are valid.
NEGATIVE_CLEANING_RULES: Dict[str, Dict[str, float]] = {
    "ammonia_mg_l": {"noise_min": -0.01},
    "bod_mg_l": {"noise_min": -0.10},
    "do_mg_l": {"noise_min": -0.15},
    "orthophosphate_mg_l": {"noise_min": -0.01},
    "nitrogen_mg_l": {"noise_min": -0.01},
    "nitrate_mg_l": {"noise_min": -0.01},
}

# Legacy alias for backward compatibility with existing tests
SDL_CLIP_COLUMNS: List[str] = list(NEGATIVE_CLEANING_RULES.keys())

CCME_WQI_CATEGORIES: List[str] = ["Poor", "Marginal", "Fair", "Good", "Excellent"]
CCME_WQI_MAP: Dict[str, int] = {cat: idx for idx, cat in enumerate(CCME_WQI_CATEGORIES)}

MISSING_VALUE_STRATEGY: Dict[str, str] = {
    "ammonia_mg_l": "Group median imputation by (country, waterbody_type) on train split.",
    "orthophosphate_mg_l": "Group median imputation by (country, waterbody_type) on train split.",
    "nitrogen_mg_l": "Group median imputation by (country, waterbody_type) on train split.",
    "nitrate_mg_l": "Group median imputation by (country, waterbody_type) on train split.",
    "bod_mg_l": "Group median imputation by (country, waterbody_type) on train split.",
    "do_mg_l": "Group median imputation by (country, waterbody_type) on train split.",
    "ph": "Group median imputation by (country, waterbody_type) on train split.",
    "temperature_c": "Group median imputation by (country, month) on train split.",
    "np_ratio": "Group median imputation by (country, waterbody_type) on train split.",
    "bod_do_ratio": "Group median imputation by (country, waterbody_type) on train split.",
    "do_deficit": "Group median imputation by (country, month) on train split.",
    "month": "Mode / median month imputation on train split.",
    "quarter": "Derived from month on train split.",
    "month_sin": "Cyclical sin transform of imputed month.",
    "month_cos": "Cyclical cos transform of imputed month.",
    "ccme_values": "Complete (continuous regression target) - no imputation.",
    "ccme_wqi": "Complete (categorical classification target) - no imputation.",
    "ccme_wqi_code": "Complete (ordinal classification target) - no imputation.",
    "country": "Complete - no imputation.",
    "area": "Complete metadata - blank strings normalized to NaN.",
    "waterbody_type": "Complete metadata - no imputation.",
    "date": "Parsed to datetime64[ns].",
}

BASE_NUMERIC_FEATURES: List[str] = [
    "ph",
    "temperature_c",
    "do_mg_l",
    "bod_mg_l",
    "ammonia_mg_l",
    "orthophosphate_mg_l",
    "nitrogen_mg_l",
    "nitrate_mg_l",
]

ENGINEERED_FEATURES: List[str] = [
    "np_ratio",
    "bod_do_ratio",
    "do_deficit",
    "month",
    "quarter",
    "month_sin",
    "month_cos",
]

# Authoritative list of 15 predictor features used across XGBoost, Random Forest, and MLP models
MODEL_INPUT_FEATURES: List[str] = BASE_NUMERIC_FEATURES + ENGINEERED_FEATURES

TARGET_REGRESSION: str = "ccme_values"
TARGET_CLASSIFICATION: str = "ccme_wqi"
TARGET_CLASSIFICATION_ORDINAL: str = "ccme_wqi_code"


@dataclass
class CleaningSummary:
    """Summary record of data cleaning pipeline steps."""

    raw_rows: int = 0
    raw_cols: int = 0
    steps: List[Dict[str, Any]] = field(default_factory=list)
    final_rows: int = 0
    final_cols: int = 0
    flags_added: List[str] = field(default_factory=list)
    output_path: Optional[str] = None
    elapsed_seconds: float = 0.0

    def log_step(
        self,
        name: str,
        rows_before: int,
        rows_after: int,
        notes: str = "",
    ) -> None:
        removed = rows_before - rows_after
        self.steps.append({
            "step": name,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "rows_removed": removed,
            "rows_removed_pct": round(removed / rows_before * 100, 4) if rows_before else 0.0,
            "notes": notes,
        })
        logger.info(
            "Step [%s]: %s -> %s rows (%s%s rows) %s",
            name,
            f"{rows_before:,}",
            f"{rows_after:,}",
            "-" if removed > 0 else "+",
            f"{abs(removed):,}",
            notes,
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)

    def print_summary(self) -> None:
        total_removed = self.raw_rows - self.final_rows
        pct_removed = total_removed / self.raw_rows * 100 if self.raw_rows else 0.0
        print("\n" + "=" * 65)
        print("Data Cleaning Pipeline Summary")
        print("=" * 65)
        print(f"Raw shape     : {self.raw_rows:,} rows x {self.raw_cols} cols")
        print(f"Cleaned shape : {self.final_rows:,} rows x {self.final_cols} cols")
        print(f"Rows dropped  : {total_removed:,} ({pct_removed:.3f}%)")
        print(f"Flags added   : {len(self.flags_added)}")
        print(f"Time taken    : {self.elapsed_seconds:.1f}s")
        if self.output_path:
            print(f"Saved to      : {self.output_path}")
        print("-" * 65)
        for s in self.steps:
            delta = s["rows_removed"]
            sign = "-" if delta > 0 else " "
            print(f"  {s['step']:<30} {s['rows_before']:>9,} -> {s['rows_after']:>9,} ({sign}{abs(delta):,})")
        print("=" * 65 + "\n")


class WaterQualityPreprocessor:
    """Cleans raw water quality records and prepares clean tabular data."""

    def __init__(
        self,
        raw_path: str,
        output_path: Optional[str] = None,
        nrows: Optional[int] = None,
    ) -> None:
        self.raw_path = raw_path
        self.output_path = output_path
        self.nrows = nrows
        self.summary = CleaningSummary()

    def run(self) -> Tuple[pd.DataFrame, CleaningSummary]:
        t0 = time.time()
        logger.info("Starting water quality cleaning pipeline")

        df = self._load()
        self.summary.raw_rows = len(df)
        self.summary.raw_cols = df.shape[1]

        df = self._standardise_column_names(df)
        df = self._cast_data_types(df)
        df = self._replace_sentinels(df)
        df = self._fix_usa_temperature_unit(df)
        df = self._clean_negative_concentrations(df)
        df = self._drop_exact_duplicates(df)
        df = self._flag_suspicious_values(df)
        df = self._apply_missing_value_strategy(df)

        self.summary.final_rows = len(df)
        self.summary.final_cols = df.shape[1]
        self.summary.elapsed_seconds = round(time.time() - t0, 2)

        df.attrs["column_mapping"] = COLUMN_NAME_MAP
        df.attrs["missing_value_strategy"] = MISSING_VALUE_STRATEGY
        df.attrs["cleaning_summary"] = self.summary.to_dataframe().to_dict("records")
        df.attrs["source_path"] = self.raw_path

        if self.output_path:
            self._save(df)
            self.summary.output_path = self.output_path

        self.summary.print_summary()
        return df, self.summary

    def _load(self) -> pd.DataFrame:
        path = self.raw_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Raw dataset not found: {path}")

        ext = Path(path).suffix.lower()
        if ext == ".parquet":
            if self.nrows is not None:
                import pyarrow.parquet as pq
                df = pq.read_table(path).slice(0, self.nrows).to_pandas()
            else:
                df = pd.read_parquet(path)
        elif ext == ".csv":
            df = pd.read_csv(path, low_memory=False, nrows=self.nrows)
        else:
            raise ValueError(f"Unsupported format '{ext}'. Expected .parquet or .csv.")

        logger.info("Loaded %s rows x %s cols from %s", f"{len(df):,}", df.shape[1], path)
        return df

    def _standardise_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        existing_map = {k: v for k, v in COLUMN_NAME_MAP.items() if k in df.columns}
        return df.rename(columns=existing_map)

    def _cast_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        rows_before = len(df)

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

        if "ccme_wqi" in df.columns:
            present_cats = [c for c in CCME_WQI_CATEGORIES if c in df["ccme_wqi"].values]
            cat_dtype = pd.CategoricalDtype(categories=present_cats, ordered=True)
            df["ccme_wqi"] = df["ccme_wqi"].astype(cat_dtype)

        # Normalize blank / null string artifacts in categorical text columns
        null_strings = {"nan", "null", "none", "na", "n/a", "nil", "", "?"}
        for col in ["country", "area", "waterbody_type"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df.loc[df[col].str.lower().isin(null_strings), col] = np.nan

        self.summary.log_step("cast_data_types", rows_before, len(df), "Casting dtypes")
        return df

    def _replace_sentinels(self, df: pd.DataFrame) -> pd.DataFrame:
        rows_before = len(df)
        total_replaced = 0

        for col in SENTINEL_COLUMNS:
            if col not in df.columns:
                continue
            mask = (df[col] <= -999.0) | df[col].isin(SENTINEL_VALUES)
            n = int(mask.sum())
            if n > 0:
                df.loc[mask, col] = np.nan
                total_replaced += n

        if "ph" in df.columns:
            # Physical pH boundaries: surface fresh/marine waters cannot exceed [0, 14]
            mask_low = df["ph"] < 0.0
            mask_high = df["ph"] > 14.0
            n_extra = int((mask_low | mask_high).sum())
            df.loc[mask_low | mask_high, "ph"] = np.nan
            total_replaced += n_extra

        if "temperature_c" in df.columns:
            # Unphysical negative temperatures below -10C (excluding valid freeze slush)
            mask_temp_low = df["temperature_c"] < -10.0
            n_temp_low = int(mask_temp_low.sum())
            if n_temp_low > 0:
                df.loc[mask_temp_low, "temperature_c"] = np.nan
                total_replaced += n_temp_low

        self.summary.log_step(
            "replace_sentinels",
            rows_before,
            len(df),
            f"{total_replaced:,} sentinels set to NaN",
        )
        return df

    def _fix_usa_temperature_unit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts historical USA river/stream temperature readings recorded in Fahrenheit
        (5,852 records from USGS stations, primarily 1948-1986 with T > 50F) to Celsius.
        """
        rows_before = len(df)
        if "temperature_c" in df.columns and "country" in df.columns:
            mask = (df["country"] == "USA") & (df["temperature_c"] > 50.0)
            n_fixed = int(mask.sum())
            if n_fixed > 0:
                df.loc[mask, "temperature_c"] = (df.loc[mask, "temperature_c"] - 32.0) * 5.0 / 9.0
            self.summary.log_step(
                "fix_usa_temperature_unit",
                rows_before,
                len(df),
                f"{n_fixed:,} US degF values converted to degC",
            )
        return df

    def _clean_negative_concentrations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans negative chemical concentrations using column-specific physical thresholds.
        Small negatives within analytical detection-limit noise are clipped to 0.0 mg/L.
        Large negatives indicating failed incubation or probe failures are set to NaN.
        """
        rows_before = len(df)
        total_clipped = 0
        total_invalid_nan = 0

        for col, rules in NEGATIVE_CLEANING_RULES.items():
            if col not in df.columns:
                continue
            noise_min = rules["noise_min"]

            # 1. Detection limit zero-calibration noise: [noise_min, 0.0) -> clip to 0.0
            noise_mask = (df[col] < 0.0) & (df[col] >= noise_min)
            n_noise = int(noise_mask.sum())
            if n_noise > 0:
                df.loc[noise_mask, col] = 0.0
                total_clipped += n_noise

            # 2. Severe negatives below noise threshold: physically invalid -> convert to NaN
            invalid_mask = df[col] < noise_min
            n_invalid = int(invalid_mask.sum())
            if n_invalid > 0:
                df.loc[invalid_mask, col] = np.nan
                total_invalid_nan += n_invalid

        self.summary.log_step(
            "clean_negative_concentrations",
            rows_before,
            len(df),
            f"{total_clipped:,} noise clipped to 0.0, {total_invalid_nan:,} unphysical set to NaN",
        )
        return df

    def _drop_exact_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops exact duplicate rows across all 14 columns (originating from Canadian ingestion merges).
        Preserves repeated measurements on different dates or at different sampling stations.
        """
        rows_before = len(df)
        df = df.drop_duplicates(keep="first")
        n_dropped = rows_before - len(df)
        self.summary.log_step(
            "drop_exact_duplicates",
            rows_before,
            len(df),
            f"{n_dropped:,} exact 14-column duplicate rows dropped",
        )
        return df

    def _flag_suspicious_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Appends non-destructive boolean audit flags for observations exceeding plausible
        environmental physical limits (without deleting valid high-pollution industrial events).
        """
        rows_before = len(df)
        flag_cols: List[str] = []
        for col, bounds in SUSPICIOUS_VALUE_RULES.items():
            if col not in df.columns:
                continue
            flag_col = f"{col}_flag_suspicious"
            mask = df[col].notna() & ((df[col] < bounds["hard_min"]) | (df[col] > bounds["hard_max"]))
            df[flag_col] = mask
            flag_cols.append(flag_col)

        self.summary.flags_added = flag_cols
        self.summary.log_step("flag_suspicious_values", rows_before, len(df), f"{len(flag_cols)} flag columns added")
        return df

    def _apply_missing_value_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        rows_before = len(df)
        for col in ["country", "waterbody_type", "ccme_values", "ccme_wqi"]:
            if col in df.columns:
                n_miss = int(df[col].isna().sum())
                if n_miss > 0:
                    logger.warning("Column '%s' has %d missing values", col, n_miss)

        self.summary.log_step("missing_value_strategy", rows_before, len(df), "Strategy logged")
        return df

    def _save(self, df: pd.DataFrame) -> None:
        out = Path(self.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(str(out), compression="snappy", index=False)
        logger.info("Saved cleaned dataset to %s (%.2f MB)", out, out.stat().st_size / (1024 * 1024))


def run_cleaning_pipeline(
    raw_path: Optional[str] = None,
    output_path: Optional[str] = None,
    nrows: Optional[int] = None,
) -> Tuple[pd.DataFrame, CleaningSummary]:
    """Runs data cleaning on the combined dataset."""
    if raw_path is None:
        raw_path = os.path.join(
            "data", "Dataset", "Parquet", "Combined Data", "Combined_dataset.parquet"
        )
    if output_path is None:
        output_path = os.path.join(
            "data", "GeneratedOutput", "02.Preprocessing&FeatureSelection", "cleaned_water_quality.parquet"
        )

    pipeline = WaterQualityPreprocessor(
        raw_path=raw_path,
        output_path=output_path,
        nrows=nrows,
    )
    return pipeline.run()


# ---------------------------------------------------------------------------
# Feature Engineering, Splitting, Imputation, and Scaling
# ---------------------------------------------------------------------------

def calculate_do_saturation(temp_c: pd.Series, formula: str = "elmore_hayes") -> pd.Series:
    """
    Calculates theoretical freshwater dissolved oxygen saturation (mg/L) at 1 atm.

    Scientific Foundations & Formulation:
      1. Elmore & Hayes (1960) / APHA Standard Methods 4500-O polynomial approximation (Default):
         DO_sat(T) = 14.652 - 0.41022*T + 0.007991*T^2 - 0.000077774*T^3
      2. Benson & Krause (1984) / USGS TWRI Book 9, Chap A6 thermodynamic standard:
         ln C* = -139.34411 + (1.575701e5 / T_K) - (6.642308e7 / T_K^2)
                 + (1.243800e10 / T_K^3) - (8.621949e11 / T_K^4)
         where T_K = T_C + 273.15.

    Supported Temperature Domain & Boundary Handling:
      - Valid liquid range: 0.0°C to 40.0°C (operational safety extension up to 45.0°C).
      - Sub-Zero Observations (-10.0°C <= T < 0.0°C, e.g. Canadian/Irish winter surface waters):
        In open natural systems, freshwater gas solubility equations describe the liquid phase.
        When surface water supercools or exists in equilibrium with frazil/slush ice, the maximum
        equilibrium concentration of dissolved oxygen in liquid water is thermodynamically bounded
        by the saturation concentration at the freezing point (0.0°C: 14.652 mg/L via APHA,
        14.621 mg/L via Benson-Krause). Extrapolating gas solubility curves into negative temperatures
        creates unphysical mathematical divergence. Therefore, valid sub-zero temperatures are
        clipped to 0.0°C for the saturation calculation, representing the physical limit at the
        ice-water phase boundary.
      - Hardware Sentinels (T < -10.0°C): Neutralized to np.nan during cleaning; returns np.nan.
      - Extreme High Temperatures (T > 45.0°C): Clipped to 45.0°C for numerical stability and flagged.
      - Missing Values (NaN): Evaluates cleanly to np.nan and is imputed during train-only imputation.
    """
    is_na = temp_c.isna()
    t = temp_c.clip(lower=0.0, upper=45.0)

    if formula == "elmore_hayes":
        sat = 14.652 - (0.41022 * t) + (0.007991 * (t ** 2)) - (0.000077774 * (t ** 3))
    elif formula == "benson_krause":
        tk = t + 273.15
        ln_c = (
            -139.34411
            + (1.575701e5 / tk)
            - (6.642308e7 / (tk ** 2))
            + (1.243800e10 / (tk ** 3))
            - (8.621949e11 / (tk ** 4))
        )
        sat = np.exp(ln_c)
    else:
        raise ValueError(
            f"Unknown DO saturation formula: '{formula}'. Choose 'elmore_hayes' or 'benson_krause'."
        )

    return sat.where(~is_na, np.nan)


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Generates domain-grounded geochemical features:
      - np_ratio: Total Nitrogen to Orthophosphate ratio (Redfield eutrophication limit)
      - bod_do_ratio: Organic loading relative to available dissolved oxygen
      - do_deficit: Signed oxygen deficit DO_sat(T) - DO (positive = deficit, negative = supersaturation)
      - month, quarter, month_sin, month_cos: Correct cyclical seasonal encoding with adjacent Jan/Dec
      - ccme_wqi_code: Ordinal target encoding (0: Poor -> 4: Excellent)
    """
    out = df.copy()
    new_features: List[str] = []

    # 1. Nutrient ratio (N:P) - Safe division, undefined when P <= 0 or inputs null
    if "nitrogen_mg_l" in out.columns and "orthophosphate_mg_l" in out.columns:
        valid_np = (
            out["nitrogen_mg_l"].notna()
            & out["orthophosphate_mg_l"].notna()
            & (out["orthophosphate_mg_l"] > 0.0)
            & (out["nitrogen_mg_l"] >= 0.0)
        )
        out["np_ratio"] = np.where(
            valid_np,
            out["nitrogen_mg_l"] / out["orthophosphate_mg_l"],
            np.nan,
        )
        new_features.append("np_ratio")

    # 2. Organic load to oxygen ratio - Safe division, undefined when DO <= 0 or inputs null
    if "bod_mg_l" in out.columns and "do_mg_l" in out.columns:
        valid_bod_do = (
            out["bod_mg_l"].notna()
            & out["do_mg_l"].notna()
            & (out["do_mg_l"] > 0.0)
            & (out["bod_mg_l"] >= 0.0)
        )
        out["bod_do_ratio"] = np.where(
            valid_bod_do,
            out["bod_mg_l"] / out["do_mg_l"],
            np.nan,
        )
        new_features.append("bod_do_ratio")

    # 3. Signed Dissolved Oxygen Deficit: DO_sat(T) - DO
    # Positive deficit = oxygen depletion due to organic decay / BOD
    # Negative deficit = super-saturation due to intense photosynthetic algal blooms
    if "do_mg_l" in out.columns and "temperature_c" in out.columns:
        valid_d = out["do_mg_l"].notna() & out["temperature_c"].notna()
        sat = calculate_do_saturation(out["temperature_c"])
        out["do_deficit"] = np.where(valid_d, sat - out["do_mg_l"], np.nan)
        new_features.append("do_deficit")

    # 4. Seasonal cyclical month encodings:
    # month_sin = sin(2 * pi * (month - 1) / 12)
    # month_cos = cos(2 * pi * (month - 1) / 12)
    # For Jan (m=1): angle=0, sin=0.0, cos=1.0. For Dec (m=12): angle=11*pi/6, sin=-0.5, cos=0.866.
    # Distance between Dec and Jan is exactly 30 degrees (pi/6 radians), ensuring smooth circular adjacency.
    if "date" in out.columns and pd.api.types.is_datetime64_any_dtype(out["date"]):
        month = out["date"].dt.month
        out["month"] = month
        out["quarter"] = out["date"].dt.quarter
        out["month_sin"] = np.where(
            month.notna(),
            np.sin(2.0 * np.pi * (month - 1.0) / 12.0),
            np.nan,
        )
        out["month_cos"] = np.where(
            month.notna(),
            np.cos(2.0 * np.pi * (month - 1.0) / 12.0),
            np.nan,
        )
        new_features.extend(["month", "quarter", "month_sin", "month_cos"])

    # 5. Ordinal target encoding for classification (0: Poor, 1: Marginal, 2: Fair, 3: Good, 4: Excellent)
    if "ccme_wqi" in out.columns:
        out["ccme_wqi_code"] = out["ccme_wqi"].map(CCME_WQI_MAP).astype("Int64")

    logger.info("Engineered %d new features: %s", len(new_features), new_features)
    return out, new_features


def split_data(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    method: str = "stratified_random",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits data into train, validation, and test partitions using one of three evaluation regimes:
      1. 'stratified_random' (Default Benchmark): 70/15/15 split stratified by ccme_wqi.
         Evaluates in-domain tabular prediction across mixed stations and years.
      2. 'country_area_group' (formerly 'spatial_group'): Group-based holdout split by country and area ('country___area').
         Guarantees that geographic areas/catchments in Train, Val, and Test are 100% mutually exclusive.
         
         Limitation & Generalization Caveat:
         The 'area' column represents a mixture of granular monitoring station codes (e.g., USGS station IDs
         in the USA, alphanumeric codes in Canada), whole waterbody names (e.g., 'Lake Sammamish' in USA),
         composite catchment strings (in Ireland), and entire bays (e.g., in China, all 45,997 records share
         'Hou Bay'). Therefore, this split evaluates generalization across distinct geographic areas and
         catchments, NOT verified individual physical sensor stations. Claims of verified station-level
         generalization must not be made.
      3. 'temporal_holdout': Time-based split (Train <= 2020, Val = 2021, Test >= 2022).
         Guarantees strictly historical training and evaluates future forecasting under decadal shift.
    """
    total = train_size + val_size + test_size
    assert math.isclose(total, 1.0, rel_tol=1e-3), "Split ratios must sum to 1.0"

    stratify_col = "ccme_wqi" if "ccme_wqi" in df.columns else None

    if method == "stratified_random":
        val_test_ratio = val_size + test_size
        train_df, holdout_df = train_test_split(
            df,
            test_size=val_test_ratio,
            random_state=random_state,
            stratify=df[stratify_col] if stratify_col else None,
        )

        test_ratio_of_holdout = test_size / val_test_ratio
        val_df, test_df = train_test_split(
            holdout_df,
            test_size=test_ratio_of_holdout,
            random_state=random_state,
            stratify=holdout_df[stratify_col] if stratify_col else None,
        )

    elif method in ("country_area_group", "spatial_group"):
        # Group by country and area identifier
        group_id = df["country"].astype(str) + "___" + df["area"].astype(str)
        val_test_ratio = val_size + test_size

        gss1 = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
        train_idx, holdout_idx = next(gss1.split(df, groups=group_id))
        train_df = df.iloc[train_idx].copy()
        holdout_df = df.iloc[holdout_idx].copy()

        test_ratio_of_holdout = test_size / val_test_ratio
        gss2 = GroupShuffleSplit(n_splits=1, train_size=(1.0 - test_ratio_of_holdout), random_state=random_state)
        val_sub_idx, test_sub_idx = next(gss2.split(holdout_df, groups=group_id.iloc[holdout_idx]))
        val_df = holdout_df.iloc[val_sub_idx].copy()
        test_df = holdout_df.iloc[test_sub_idx].copy()

    elif method == "temporal_holdout":
        assert "date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["date"]), (
            "Temporal split requires parsed datetime 'date' column."
        )
        years = df["date"].dt.year
        train_df = df[years <= 2020].copy()
        val_df = df[years == 2021].copy()
        test_df = df[years >= 2022].copy()

    else:
        raise ValueError(
            f"Unknown split method: {method}. Choose 'stratified_random', 'country_area_group' (or 'spatial_group'), or 'temporal_holdout'."
        )

    logger.info(
        "Split data (%s): Train=%s (%.1f%%), Val=%s (%.1f%%), Test=%s (%.1f%%)",
        method,
        f"{len(train_df):,}",
        len(train_df) / len(df) * 100,
        f"{len(val_df):,}",
        len(val_df) / len(df) * 100,
        f"{len(test_df):,}",
        len(test_df) / len(df) * 100,
    )
    return train_df.copy(), val_df.copy(), test_df.copy()


def impute_missing_values(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Fits domain-informed median imputation strictly on the training set and transforms train, val, and test:
      - Chemical features and nutrient ratios: Group medians by (country, waterbody_type).
      - Temperature and seasonal variables: Group medians by (country, month).
      - Hierarchical fallback: Group median -> Country median -> Global train median -> 0.0.
    Target variables (ccme_values, ccme_wqi, ccme_wqi_code) are NEVER used in imputation fitting.
    """
    train_out = train_df.copy()
    val_out = val_df.copy()
    test_out = test_df.copy()

    # Learn global train medians as ultimate fallback
    global_medians: Dict[str, float] = {}
    for col in feature_cols:
        if col in train_df.columns and pd.api.types.is_numeric_dtype(train_df[col]):
            med = train_df[col].median(skipna=True)
            global_medians[col] = float(med) if pd.notna(med) else 0.0

    # Learn country-level medians as intermediate fallback
    country_medians: Dict[str, Dict[str, float]] = {}
    if "country" in train_df.columns:
        for col in feature_cols:
            if col in train_df.columns and pd.api.types.is_numeric_dtype(train_df[col]):
                country_medians[col] = train_df.groupby("country", observed=False)[col].median().to_dict()

    # Primary group keys
    chem_group_cols = [c for c in ["country", "waterbody_type"] if c in train_df.columns]
    temp_group_cols = [c for c in ["country", "month"] if c in train_df.columns]

    def _get_group_key(df: pd.DataFrame, cols: List[str]) -> pd.Series:
        if len(cols) == 2:
            return df[cols[0]].astype(str) + "___" + df[cols[1]].astype(str)
        elif len(cols) == 1:
            return df[cols[0]].astype(str)
        return pd.Series("", index=df.index)

    chem_train_keys = _get_group_key(train_df, chem_group_cols)
    chem_val_keys = _get_group_key(val_df, chem_group_cols)
    chem_test_keys = _get_group_key(test_df, chem_group_cols)

    temp_train_keys = _get_group_key(train_df, temp_group_cols)
    temp_val_keys = _get_group_key(val_df, temp_group_cols)
    temp_test_keys = _get_group_key(test_df, temp_group_cols)

    seasonal_features = {"temperature_c", "do_deficit", "month", "quarter", "month_sin", "month_cos"}

    for col in feature_cols:
        if col not in train_out.columns or not pd.api.types.is_numeric_dtype(train_out[col]):
            continue

        g_med = global_medians.get(col, 0.0)
        c_med_map = country_medians.get(col, {})

        is_seasonal = col in seasonal_features and len(temp_group_cols) > 0

        if is_seasonal:
            group_keys_train = temp_train_keys
            group_keys_val = temp_val_keys
            group_keys_test = temp_test_keys
        else:
            group_keys_train = chem_train_keys
            group_keys_val = chem_val_keys
            group_keys_test = chem_test_keys

        # Compute group medians strictly on training split
        group_medians = train_df.groupby(group_keys_train, observed=False)[col].median().to_dict()

        for partition_df, keys in [(train_out, group_keys_train), (val_out, group_keys_val), (test_out, group_keys_test)]:
            # 1. Fill with group median
            partition_df[col] = partition_df[col].fillna(keys.map(group_medians))
            # 2. Fallback to country median
            if "country" in partition_df.columns and c_med_map:
                partition_df[col] = partition_df[col].fillna(partition_df["country"].map(c_med_map))
            # 3. Fallback to global train median
            partition_df[col] = partition_df[col].fillna(g_med)

    imputer_info = {
        "global_medians": global_medians,
        "chem_group_cols": chem_group_cols,
        "temp_group_cols": temp_group_cols,
        "feature_cols": feature_cols,
    }
    return train_out, val_out, test_out, imputer_info


def scale_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fits StandardScaler strictly on the training set numerical input features and
    transforms train, validation, and test sets.
    Target variables (ccme_values, ccme_wqi, ccme_wqi_code) are NEVER scaled.
    """
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])

    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    test_scaled = test_df.copy()

    train_scaled[feature_cols] = scaler.transform(train_df[feature_cols])
    val_scaled[feature_cols] = scaler.transform(val_df[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])

    return train_scaled, val_scaled, test_scaled, scaler


def prepare_training_datasets(
    cleaned_df: Optional[pd.DataFrame] = None,
    cleaned_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    split_method: str = "stratified_random",
    sample_size: int = 50000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    End-to-end dataset preparation pipeline:
      1. Feature Engineering (nutrient ratios, oxygen deficit, cyclical seasonal encoding)
      2. Splitting (stratified_random / spatial_group / temporal_holdout)
      3. Training-only Group Median Imputation (with hierarchical fallback)
      4. Training-only StandardScaler fitting for Deep Learning (MLP)
      5. Exporting unscaled and scaled Parquet partitions and lightweight samples
      6. Generating comprehensive provenance and data audit metadata
    """
    if output_dir is None:
        output_dir = os.path.join("data", "GeneratedOutput", "02.Preprocessing&FeatureSelection")
    os.makedirs(output_dir, exist_ok=True)

    if cleaned_df is None:
        if cleaned_path is None:
            cleaned_path = os.path.join(output_dir, "cleaned_water_quality.parquet")
        if os.path.exists(cleaned_path):
            logger.info("Loading cleaned data from %s", cleaned_path)
            cleaned_df = pd.read_parquet(cleaned_path)
        else:
            logger.info("Running cleaning pipeline first...")
            cleaned_df, _ = run_cleaning_pipeline(output_path=cleaned_path)

    # 1. Feature Engineering
    df_feat, new_cols = engineer_features(cleaned_df)

    # Enforce exact model input feature list
    modeling_features = [f for f in MODEL_INPUT_FEATURES if f in df_feat.columns]

    # 2. Splitting
    train_df, val_df, test_df = split_data(
        df_feat,
        train_size=0.70,
        val_size=0.15,
        test_size=0.15,
        method=split_method,
        random_state=random_state,
    )

    # 3. Imputation (strictly fitted on train)
    train_imp, val_imp, test_imp, imputer_info = impute_missing_values(
        train_df, val_df, test_df, feature_cols=modeling_features
    )

    # 4. Scaling (for MLP / Deep Learning)
    train_scaled, val_scaled, test_scaled, scaler = scale_features(
        train_imp, val_imp, test_imp, feature_cols=modeling_features
    )

    # 5. Save Parquet datasets
    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "val.parquet")
    test_path = os.path.join(output_dir, "test.parquet")
    train_scaled_path = os.path.join(output_dir, "train_scaled.parquet")
    val_scaled_path = os.path.join(output_dir, "val_scaled.parquet")
    test_scaled_path = os.path.join(output_dir, "test_scaled.parquet")

    train_imp.to_parquet(train_path, compression="snappy", index=False)
    val_imp.to_parquet(val_path, compression="snappy", index=False)
    test_imp.to_parquet(test_path, compression="snappy", index=False)
    train_scaled.to_parquet(train_scaled_path, compression="snappy", index=False)
    val_scaled.to_parquet(val_scaled_path, compression="snappy", index=False)
    test_scaled.to_parquet(test_scaled_path, compression="snappy", index=False)

    # Save lightweight sample files for rapid experimentation (e.g. CI / local testing)
    train_sample_path = os.path.join(output_dir, "train_sample.parquet")
    val_sample_path = os.path.join(output_dir, "val_sample.parquet")
    test_sample_path = os.path.join(output_dir, "test_sample.parquet")

    if sample_size and len(train_imp) > sample_size:
        train_imp.sample(n=sample_size, random_state=random_state).to_parquet(
            train_sample_path, compression="snappy", index=False
        )
        val_imp.sample(n=min(len(val_imp), sample_size // 4), random_state=random_state).to_parquet(
            val_sample_path, compression="snappy", index=False
        )
        test_imp.sample(n=min(len(test_imp), sample_size // 4), random_state=random_state).to_parquet(
            test_sample_path, compression="snappy", index=False
        )

    # 6. Comprehensive Metadata
    metadata = {
        "metadata_version": "2.0.0",
        "dataset_source": "data/Dataset/Parquet/Combined Data/Combined_dataset.parquet",
        "source_filename": "Combined_dataset.parquet",
        "original_row_count": 4297895,
        "cleaned_row_count": len(cleaned_df),
        "rows_removed": 4297895 - len(cleaned_df),
        "exact_duplicates_removed": 10005,
        "usa_temperature_degf_converted": 5852,
        "sentinels_neutralized": 84,
        "split_methodology": split_method,
        "random_seed": random_state,
        "split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "split_counts": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "sample_sizes": {
            "train_sample": min(len(train_df), sample_size),
            "val_sample": min(len(val_df), sample_size // 4),
            "test_sample": min(len(test_df), sample_size // 4),
        },
        "input_features_ordered": modeling_features,
        "input_feature_count": len(modeling_features),
        "base_features": BASE_NUMERIC_FEATURES,
        "engineered_features": ENGINEERED_FEATURES,
        "target_definitions": {
            "regression_target": TARGET_REGRESSION,
            "classification_target": TARGET_CLASSIFICATION,
            "classification_target_ordinal": TARGET_CLASSIFICATION_ORDINAL,
        },
        "target_class_mapping": CCME_WQI_MAP,
        "target_class_order": CCME_WQI_CATEGORIES,
        "imputation_strategy": {
            "chemical_features": "Group median by (country, waterbody_type) strictly on train split",
            "seasonal_features": "Group median by (country, month) strictly on train split",
            "fallback_hierarchy": ["group_median", "country_median", "global_train_median", 0.0],
            "global_medians_fitted_on_train": imputer_info["global_medians"],
        },
        "scaler_parameters": {
            "feature_names": modeling_features,
            "mean": dict(zip(modeling_features, scaler.mean_.tolist())),
            "scale": dict(zip(modeling_features, scaler.scale_.tolist())),
        },
        "leakage_safeguards": [
            "Targets (ccme_values, ccme_wqi, ccme_wqi_code) strictly excluded from MODEL_INPUT_FEATURES",
            "Imputer statistics fitted strictly on the 70% train split and applied to val/test without refitting",
            "StandardScaler fitted strictly on the 70% train split and applied to val/test without refitting",
            "Exact 14-column duplicate rows removed prior to splitting to prevent memorization",
            "Alternative country_area_group split guarantees mutually exclusive country-area geographic clusters between train, val, and test (not claiming verified station-level generalization due to area heterogeneity)",
        ],
        "output_artifacts": {
            "cleaned_data": "data/GeneratedOutput/02.Preprocessing&FeatureSelection/cleaned_water_quality.parquet",
            "train_unscaled": "data/GeneratedOutput/02.Preprocessing&FeatureSelection/train.parquet",
            "val_unscaled": "data/GeneratedOutput/02.Preprocessing&FeatureSelection/val.parquet",
            "test_unscaled": "data/GeneratedOutput/02.Preprocessing&FeatureSelection/test.parquet",
            "train_scaled": "data/GeneratedOutput/02.Preprocessing&FeatureSelection/train_scaled.parquet",
            "val_scaled": "data/GeneratedOutput/02.Preprocessing&FeatureSelection/val_scaled.parquet",
            "test_scaled": "data/GeneratedOutput/02.Preprocessing&FeatureSelection/test_scaled.parquet",
            "train_sample": "data/GeneratedOutput/02.Preprocessing&FeatureSelection/train_sample.parquet",
            "val_sample": "data/GeneratedOutput/02.Preprocessing&FeatureSelection/val_sample.parquet",
            "test_sample": "data/GeneratedOutput/02.Preprocessing&FeatureSelection/test_sample.parquet",
            "metadata_file": "data/GeneratedOutput/02.Preprocessing&FeatureSelection/preprocessing_metadata.json",
        },
    }

    meta_path = os.path.join(output_dir, "preprocessing_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Successfully generated all training datasets in %s", output_dir)
    return {
        "train": train_imp,
        "val": val_imp,
        "test": test_imp,
        "train_scaled": train_scaled,
        "val_scaled": val_scaled,
        "test_scaled": test_scaled,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Validation helpers (used in tests and validation scripts)
# ---------------------------------------------------------------------------

def assert_schema_preserved(df: pd.DataFrame) -> None:
    required_numeric = [
        "ammonia_mg_l", "bod_mg_l", "do_mg_l", "orthophosphate_mg_l",
        "ph", "temperature_c", "nitrogen_mg_l", "nitrate_mg_l", "ccme_values",
    ]
    required_meta = ["country", "area", "waterbody_type", "date", "ccme_wqi"]

    for col in required_numeric:
        assert col in df.columns, f"Missing numeric column: {col}"
        assert pd.api.types.is_numeric_dtype(df[col]), f"Column '{col}' should be numeric"

    for col in required_meta:
        assert col in df.columns, f"Missing metadata column: {col}"

    assert pd.api.types.is_datetime64_any_dtype(df["date"]), "'date' should be datetime64"
    assert hasattr(df["ccme_wqi"], "cat"), "'ccme_wqi' should be Categorical"
    assert df["ccme_wqi"].cat.ordered, "'ccme_wqi' Categorical must be ordered"


def assert_no_sentinels(df: pd.DataFrame) -> None:
    for col in ["ph", "temperature_c", "do_mg_l"]:
        if col in df.columns:
            n = int((df[col] <= -999.0).sum())
            assert n == 0, f"Column '{col}' contains sentinel values <= -999"
    if "ph" in df.columns:
        n_high = int((df["ph"] > 14.0).sum())
        n_low = int((df["ph"] < 0.0).sum())
        assert n_high == 0, f"'ph' contains {n_high} values > 14.0"
        assert n_low == 0, f"'ph' contains {n_low} values < 0.0"


def assert_no_sdl_negatives(df: pd.DataFrame) -> None:
    for col in NEGATIVE_CLEANING_RULES:
        if col in df.columns:
            n_neg = int((df[col] < 0.0).sum())
            assert n_neg == 0, f"Column '{col}' has {n_neg} negative values after cleaning"


def assert_no_exact_duplicates(df: pd.DataFrame) -> None:
    assert df.duplicated().sum() == 0, "DataFrame contains duplicate rows"


def assert_targets_complete(df: pd.DataFrame) -> None:
    for col in ["ccme_values", "ccme_wqi"]:
        if col in df.columns:
            assert df[col].isna().sum() == 0, f"Target '{col}' has missing values"


def assert_flag_columns_present(df: pd.DataFrame) -> None:
    for col in SUSPICIOUS_VALUE_RULES:
        flag_col = f"{col}_flag_suspicious"
        assert flag_col in df.columns, f"Expected flag column '{flag_col}' missing"
        assert df[flag_col].dtype == bool, f"Flag column '{flag_col}' should be bool"


def assert_usa_temperatures_in_celsius_range(df: pd.DataFrame) -> None:
    if "temperature_c" in df.columns and "country" in df.columns:
        mask = (df["country"] == "USA") & (df["temperature_c"] > 50.0)
        assert mask.sum() == 0, "USA temperatures still > 50C"


def assert_raw_unmodified(raw_df_before: pd.DataFrame, raw_df_after: pd.DataFrame) -> None:
    assert raw_df_before.shape == raw_df_after.shape
    pd.testing.assert_frame_equal(raw_df_before, raw_df_after)


def generate_data_quality_summary(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> pd.DataFrame:
    rows_raw = len(df_raw)
    rows_clean = len(df_clean)
    records = []
    for raw_col, clean_col in COLUMN_NAME_MAP.items():
        if raw_col in df_raw.columns and clean_col in df_clean.columns:
            r_null = int(df_raw[raw_col].isna().sum())
            c_null = int(df_clean[clean_col].isna().sum())
            records.append({
                "column_raw": raw_col,
                "column_clean": clean_col,
                "raw_null_count": r_null,
                "raw_null_pct": round(r_null / rows_raw * 100, 3) if rows_raw else 0.0,
                "clean_null_count": c_null,
                "clean_null_pct": round(c_null / rows_clean * 100, 3) if rows_clean else 0.0,
                "null_delta": c_null - r_null,
            })
    summary_df = pd.DataFrame(records)
    summary_df.attrs["row_counts"] = {"raw": rows_raw, "clean": rows_clean, "removed": rows_raw - rows_clean}
    summary_df.attrs["duplicate_counts"] = {
        "raw": int(df_raw.duplicated().sum()),
        "clean": int(df_clean.duplicated().sum()),
    }
    return summary_df


def load_training_data(
    data_path: str,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "ccme_values",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Authoritative training data loader for Stage 03 Model Training.

    Ensures that downstream modeling scripts (tree models, neural networks) load ONLY
    the authoritative 15 input features, that target variables are strictly excluded
    from the input matrix, and that no NaNs or Infs enter the model.

    Args:
        data_path: Path to the Parquet dataset (e.g. 'train.parquet' or 'train_scaled.parquet').
        feature_cols: List of feature names. Defaults to MODEL_INPUT_FEATURES (15 features).
        target_col: Target column to extract ('ccme_values' for regression, 'ccme_wqi' or 'ccme_wqi_code' for classification).

    Returns:
        X: numpy.ndarray of shape (N, 15) with float64 data.
        y: numpy.ndarray of shape (N,) with target labels.
        feature_names: List of feature names in exact column order.
    """
    if feature_cols is None:
        feature_cols = MODEL_INPUT_FEATURES.copy()

    # Critical leak check: Never allow target in input features
    for forbidden_target in [TARGET_REGRESSION, TARGET_CLASSIFICATION, TARGET_CLASSIFICATION_ORDINAL]:
        if forbidden_target in feature_cols:
            raise ValueError(
                f"CRITICAL LEAKAGE DETECTED: Target '{forbidden_target}' cannot be included in input feature_cols!"
            )

    df = pd.read_parquet(data_path)

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required input features in {data_path}: {missing_cols}")

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in {data_path}")

    X_df = df[feature_cols]
    y_ser = df[target_col]

    nan_count = int(X_df.isna().sum().sum())
    if nan_count > 0:
        raise ValueError(f"Input feature matrix contains {nan_count} NaN values in {data_path}")

    inf_count = int(np.isinf(X_df.select_dtypes(include=np.number).values).sum())
    if inf_count > 0:
        raise ValueError(f"Input feature matrix contains {inf_count} infinite values in {data_path}")

    X = X_df.to_numpy(dtype=np.float64)
    y = y_ser.to_numpy()

    return X, y, feature_cols


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Water Quality Preprocessing and Dataset Preparation")
    parser.add_argument("--input", type=str, default=None, help="Path to raw dataset")
    parser.add_argument("--output", type=str, default=None, help="Output path for cleaned parquet")
    parser.add_argument("--prepare-splits", action="store_true", help="Generate train/val/test splits")
    parser.add_argument(
        "--split-method",
        type=str,
        default="stratified_random",
        choices=["stratified_random", "country_area_group", "spatial_group", "temporal_holdout"],
        help="Splitting methodology",
    )
    parser.add_argument("--nrows", type=int, default=None, help="Limit rows loaded")
    args = parser.parse_args()

    cleaned, _ = run_cleaning_pipeline(
        raw_path=args.input,
        output_path=args.output,
        nrows=args.nrows,
    )

    if args.prepare_splits:
        prepare_training_datasets(cleaned_df=cleaned, split_method=args.split_method)
