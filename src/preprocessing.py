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
from sklearn.model_selection import train_test_split
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

# Sensor error sentinels found during data audit
SENTINEL_VALUES: List[float] = [-999999.0, -9999.0, -999.0, -596.0, -99.9, -80.0]
SENTINEL_COLUMNS: List[str] = ["ph", "temperature_c", "do_mg_l"]

# Plausible domain bounds used for quality flagging
SUSPICIOUS_VALUE_RULES: Dict[str, Dict[str, float]] = {
    "ph": {"hard_min": 0.0, "hard_max": 14.0},
    "temperature_c": {"hard_min": -5.0, "hard_max": 98.0},
    "ammonia_mg_l": {"hard_min": 0.0, "hard_max": 500.0},
    "bod_mg_l": {"hard_min": 0.0, "hard_max": 1000.0},
    "do_mg_l": {"hard_min": 0.0, "hard_max": 30.0},
    "orthophosphate_mg_l": {"hard_min": 0.0, "hard_max": 200.0},
    "nitrogen_mg_l": {"hard_min": 0.0, "hard_max": 500.0},
    "nitrate_mg_l": {"hard_min": 0.0, "hard_max": 1000.0},
    "ccme_values": {"hard_min": 0.0, "hard_max": 100.0},
}

# Sub-detection limit negative clipping
SDL_CLIP_COLUMNS: List[str] = [
    "ammonia_mg_l",
    "bod_mg_l",
    "do_mg_l",
    "orthophosphate_mg_l",
    "nitrogen_mg_l",
    "nitrate_mg_l",
]

CCME_WQI_CATEGORIES: List[str] = ["Poor", "Marginal", "Fair", "Good", "Excellent"]
CCME_WQI_MAP: Dict[str, int] = {cat: idx for idx, cat in enumerate(CCME_WQI_CATEGORIES)}

MISSING_VALUE_STRATEGY: Dict[str, str] = {
    "ammonia_mg_l": "Group median imputation by (country, waterbody_type) on train split.",
    "orthophosphate_mg_l": "Group median imputation by (country, waterbody_type) on train split.",
    "nitrogen_mg_l": "Group median imputation by (country, waterbody_type) on train split.",
    "nitrate_mg_l": "Group median imputation by (country, waterbody_type) on train split.",
    "bod_mg_l": "Group median imputation by (country, waterbody_type) on train split.",
    "do_mg_l": "Group median or temperature-informed oxygen saturation imputation on train split.",
    "ph": "Group median imputation by (country, waterbody_type) on train split.",
    "temperature_c": "Group median imputation by (country, month) on train split.",
    "ccme_values": "Complete (target) - no imputation.",
    "ccme_wqi": "Complete (target) - no imputation.",
    "country": "Complete - no imputation.",
    "area": "Complete - no imputation.",
    "waterbody_type": "Complete - no imputation.",
    "date": "Complete - parsed to datetime.",
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
        df = self._clip_sub_detection_limit_negatives(df)
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

        # Normalize blank / null string artifacts
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
            mask_low = df["ph"] == -1.0
            mask_high = df["ph"] > 14.0
            n_extra = int(mask_low.sum() + mask_high.sum())
            df.loc[mask_low | mask_high, "ph"] = np.nan
            total_replaced += n_extra

        self.summary.log_step(
            "replace_sentinels",
            rows_before,
            len(df),
            f"{total_replaced:,} sentinels set to NaN",
        )
        return df

    def _fix_usa_temperature_unit(self, df: pd.DataFrame) -> pd.DataFrame:
        # Convert US readings from degF to degC if above 50
        rows_before = len(df)
        if "temperature_c" in df.columns and "country" in df.columns:
            mask = (df["country"] == "USA") & (df["temperature_c"] > 50)
            n_fixed = int(mask.sum())
            if n_fixed > 0:
                df.loc[mask, "temperature_c"] = (df.loc[mask, "temperature_c"] - 32) * 5 / 9
            self.summary.log_step(
                "fix_usa_temperature_unit",
                rows_before,
                len(df),
                f"{n_fixed:,} US degF values converted to degC",
            )
        return df

    def _clip_sub_detection_limit_negatives(self, df: pd.DataFrame) -> pd.DataFrame:
        # Clip instrument noise in [-5, 0) concentration values to 0.0
        rows_before = len(df)
        total_clipped = 0
        for col in SDL_CLIP_COLUMNS:
            if col in df.columns:
                mask = (df[col] < 0.0) & (df[col] >= -5.0)
                total_clipped += int(mask.sum())
                df.loc[mask, col] = 0.0

        self.summary.log_step(
            "clip_sdl_negatives",
            rows_before,
            len(df),
            f"{total_clipped:,} values clipped to 0.0",
        )
        return df

    def _drop_exact_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        rows_before = len(df)
        df = df.drop_duplicates(keep="first")
        n_dropped = rows_before - len(df)
        self.summary.log_step("drop_exact_duplicates", rows_before, len(df), f"{n_dropped:,} duplicates dropped")
        return df

    def _flag_suspicious_values(self, df: pd.DataFrame) -> pd.DataFrame:
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
# Feature Engineering, Splitting, Imputation, and MLP Scaling
# ---------------------------------------------------------------------------

def calculate_do_saturation(temp_c: pd.Series) -> pd.Series:
    """Calculates theoretical oxygen saturation concentration at temperature T."""
    # Polynomial approximation for DO saturation in freshwater (mg/L)
    t = temp_c.clip(lower=0.0, upper=45.0)
    return 14.652 - (0.41022 * t) + (0.007991 * (t ** 2)) - (0.000077774 * (t ** 3))


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Generates nutrient ratios, oxygen deficit, and cyclical seasonal features."""
    out = df.copy()
    new_features: List[str] = []

    # Nutrient ratio (N:P)
    if "nitrogen_mg_l" in out.columns and "orthophosphate_mg_l" in out.columns:
        out["np_ratio"] = out["nitrogen_mg_l"] / (out["orthophosphate_mg_l"] + 1e-4)
        new_features.append("np_ratio")

    # Organic load to oxygen ratio
    if "bod_mg_l" in out.columns and "do_mg_l" in out.columns:
        out["bod_do_ratio"] = out["bod_mg_l"] / (out["do_mg_l"] + 1e-4)
        new_features.append("bod_do_ratio")

    # Oxygen deficit
    if "do_mg_l" in out.columns and "temperature_c" in out.columns:
        sat = calculate_do_saturation(out["temperature_c"])
        out["do_deficit"] = (sat - out["do_mg_l"]).clip(lower=0.0)
        new_features.append("do_deficit")

    # Seasonal cyclical date features
    if "date" in out.columns and pd.api.types.is_datetime64_any_dtype(out["date"]):
        month = out["date"].dt.month.fillna(6)
        out["month"] = month
        out["quarter"] = out["date"].dt.quarter.fillna(2)
        out["month_sin"] = np.sin(2 * np.pi * month / 12)
        out["month_cos"] = np.cos(2 * np.pi * month / 12)
        new_features.extend(["month", "quarter", "month_sin", "month_cos"])

    # Categorical target code for classification (0: Poor -> 4: Excellent)
    if "ccme_wqi" in out.columns:
        out["ccme_wqi_code"] = out["ccme_wqi"].map(CCME_WQI_MAP).astype("Int64")

    logger.info("Engineered %d new features: %s", len(new_features), new_features)
    return out, new_features


def split_data(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits data into train (70%), val (15%), and test (15%) stratified by target."""
    total = train_size + val_size + test_size
    assert math.isclose(total, 1.0, rel_tol=1e-3), "Split ratios must sum to 1.0"

    stratify_col = "ccme_wqi" if "ccme_wqi" in df.columns else None

    # First split off test set
    val_test_ratio = val_size + test_size
    train_df, holdout_df = train_test_split(
        df,
        test_size=val_test_ratio,
        random_state=random_state,
        stratify=df[stratify_col] if stratify_col else None,
    )

    # Next split holdout into val and test
    test_ratio_of_holdout = test_size / val_test_ratio
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=test_ratio_of_holdout,
        random_state=random_state,
        stratify=holdout_df[stratify_col] if stratify_col else None,
    )

    logger.info(
        "Split data: Train=%s (%.1f%%), Val=%s (%.1f%%), Test=%s (%.1f%%)",
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
    """Fits group medians on train split and applies to train, val, and test."""
    train_out = train_df.copy()
    val_out = val_df.copy()
    test_out = test_df.copy()

    group_cols = [c for c in ["country", "waterbody_type"] if c in train_df.columns]

    global_medians: Dict[str, float] = {}
    for col in feature_cols:
        if col in train_df.columns and pd.api.types.is_numeric_dtype(train_df[col]):
            global_medians[col] = float(train_df[col].median(skipna=True))

    def _get_group_key(df: pd.DataFrame) -> pd.Series:
        if len(group_cols) == 2:
            return df[group_cols[0]].astype(str) + "___" + df[group_cols[1]].astype(str)
        elif len(group_cols) == 1:
            return df[group_cols[0]].astype(str)
        return pd.Series("", index=df.index)

    train_keys = _get_group_key(train_df)
    val_keys = _get_group_key(val_df)
    test_keys = _get_group_key(test_df)

    for col in feature_cols:
        if col not in train_out.columns or not pd.api.types.is_numeric_dtype(train_out[col]):
            continue

        g_med = global_medians.get(col, 0.0)

        # Compute group medians on train split using fast vectorized groupby
        if group_cols:
            group_medians = train_df.groupby(train_keys, observed=False)[col].median().to_dict()
            train_out[col] = train_out[col].fillna(train_keys.map(group_medians)).fillna(g_med)
            val_out[col] = val_out[col].fillna(val_keys.map(group_medians)).fillna(g_med)
            test_out[col] = test_out[col].fillna(test_keys.map(group_medians)).fillna(g_med)
        else:
            train_out[col] = train_out[col].fillna(g_med)
            val_out[col] = val_out[col].fillna(g_med)
            test_out[col] = test_out[col].fillna(g_med)

    imputer_info = {
        "global_medians": global_medians,
        "group_cols": group_cols,
        "feature_cols": feature_cols,
    }
    return train_out, val_out, test_out, imputer_info


def scale_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fits StandardScaler on train numerical features and transforms train/val/test."""
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
    sample_size: int = 50000,
) -> Dict[str, Any]:
    """Full preparation pipeline: feature engineering, splitting, imputation, and scaling."""
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

    all_numeric_features = [
        col for col in (BASE_NUMERIC_FEATURES + new_cols)
        if col in df_feat.columns and pd.api.types.is_numeric_dtype(df_feat[col])
    ]

    # 2. Stratified Splitting
    train_df, val_df, test_df = split_data(df_feat, train_size=0.70, val_size=0.15, test_size=0.15)

    # 3. Imputation (strictly fitted on train)
    train_imp, val_imp, test_imp, imputer_info = impute_missing_values(
        train_df, val_df, test_df, feature_cols=all_numeric_features
    )

    # 4. Scaling (for MLP / Deep Learning)
    train_scaled, val_scaled, test_scaled, scaler = scale_features(
        train_imp, val_imp, test_imp, feature_cols=all_numeric_features
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

    # Save lightweight sample files for rapid experimentation
    if sample_size and len(train_imp) > sample_size:
        train_imp.sample(n=sample_size, random_state=42).to_parquet(
            os.path.join(output_dir, "train_sample.parquet"), compression="snappy", index=False
        )
        val_imp.sample(n=min(len(val_imp), sample_size // 4), random_state=42).to_parquet(
            os.path.join(output_dir, "val_sample.parquet"), compression="snappy", index=False
        )
        test_imp.sample(n=min(len(test_imp), sample_size // 4), random_state=42).to_parquet(
            os.path.join(output_dir, "test_sample.parquet"), compression="snappy", index=False
        )

    # 6. Metadata summary
    metadata = {
        "total_rows": len(cleaned_df),
        "split_counts": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "feature_columns": all_numeric_features,
        "base_features": BASE_NUMERIC_FEATURES,
        "engineered_features": new_cols,
        "targets": {
            "continuous": "ccme_values",
            "categorical": "ccme_wqi",
            "categorical_code": "ccme_wqi_code",
        },
        "target_classes": CCME_WQI_CATEGORIES,
        "scaler": {
            "mean": dict(zip(all_numeric_features, scaler.mean_.tolist())),
            "scale": dict(zip(all_numeric_features, scaler.scale_.tolist())),
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
            n = int((df[col] <= -999).sum())
            assert n == 0, f"Column '{col}' contains sentinel values <= -999"
    if "ph" in df.columns:
        n_high = int((df["ph"] > 14).sum())
        assert n_high == 0, f"'ph' contains {n_high} values > 14.0"


def assert_no_sdl_negatives(df: pd.DataFrame) -> None:
    for col in SDL_CLIP_COLUMNS:
        if col in df.columns:
            n_neg = int((df[col] < 0).sum())
            assert n_neg == 0, f"Column '{col}' has {n_neg} negative values after clipping"


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
        mask = (df["country"] == "USA") & (df["temperature_c"] > 50)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Water Quality Preprocessing and Dataset Preparation")
    parser.add_argument("--input", type=str, default=None, help="Path to raw dataset")
    parser.add_argument("--output", type=str, default=None, help="Output path for cleaned parquet")
    parser.add_argument("--prepare-splits", action="store_true", help="Generate train/val/test splits")
    parser.add_argument("--nrows", type=int, default=None, help="Limit rows loaded")
    args = parser.parse_args()

    cleaned, _ = run_cleaning_pipeline(
        raw_path=args.input,
        output_path=args.output,
        nrows=args.nrows,
    )

    if args.prepare_splits:
        prepare_training_datasets(cleaned_df=cleaned)
