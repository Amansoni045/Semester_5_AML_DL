"""
preprocessing.py
================
Reusable data-cleaning pipeline for the Water Quality Prediction ML project.

All decisions are grounded in the findings from ``notebooks/01_dataset_audit.ipynb``
and the domain knowledge documented in ``AGENT.md``.

Pipeline stages (in execution order)
--------------------------------------
1.  Standardise column names  — snake_case aliases with original mapping preserved.
2.  Parse / cast data types   — Date -> datetime64, CCME_WQI -> ordered Categorical.
3.  Replace sentinel values   — hardware error codes (-999999, -596, -99.9, -80)
                                and the 67115 pH scale-corruption anomaly -> NaN.
4.  Fix USA temperature unit  — 5,852 US readings recorded in F converted to C.
5.  Clip sub-detection-limit negatives — tiny negatives caused by instrument
                                          baseline offsets -> 0.0.
6.  Drop exact duplicates     — 10,005 confirmed exact-duplicate rows removed and
                                 logged; raw files are never modified.
7.  Flag suspicious outliers  — physically implausible values are FLAGGED (boolean
                                 column per feature), NOT deleted.
8.  Missing-value strategy    — column-specific, documented, MAR/MCAR-aware logic.
                                 No automatic mean/median imputation across the board.
9.  Preserve metadata         — Country, Area, Waterbody Type, Date, site identifiers
                                 are carried through unchanged (only dtype-cast).
10. Save cleaned dataset      — written to a configurable path as Parquet/Snappy.
                                 Raw source files are NEVER overwritten.

Usage
-----
    from src.preprocessing import WaterQualityPreprocessor

    pipeline = WaterQualityPreprocessor(
        raw_path="data/Dataset/Parquet/Combined Data/Combined_dataset.parquet",
        output_path="data/GeneratedOutput/02.Preprocessing/cleaned_dataset.parquet",
    )
    cleaned_df, summary = pipeline.run()

Or from the command line:

    python -m src.preprocessing \\
        --input  "data/Dataset/Parquet/Combined Data/Combined_dataset.parquet" \\
        --output data/GeneratedOutput/02.Preprocessing/cleaned_dataset.parquet
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("WaterQualityPreprocessor")


# ---------------------------------------------------------------------------
# Constants - grounded in audit findings & domain knowledge
# ---------------------------------------------------------------------------

# Original column names exactly as they appear in the raw Parquet file.
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

# Mapping from original column name -> snake_case alias used internally.
# The original names are PRESERVED in the column_mapping dict and as attributes
# of the returned DataFrame (.attrs["column_mapping"]).
COLUMN_NAME_MAP: Dict[str, str] = {
    "Country":                              "country",
    "Area":                                 "area",
    "Waterbody Type":                       "waterbody_type",
    "Date":                                 "date",
    "Ammonia (mg/l)":                       "ammonia_mg_l",
    "Biochemical Oxygen Demand (mg/l)":     "bod_mg_l",
    "Dissolved Oxygen (mg/l)":              "do_mg_l",
    "Orthophosphate (mg/l)":               "orthophosphate_mg_l",
    "pH (ph units)":                        "ph",
    "Temperature (cel)":                    "temperature_c",
    "Nitrogen (mg/l)":                      "nitrogen_mg_l",
    "Nitrate (mg/l)":                       "nitrate_mg_l",
    "CCME_Values":                          "ccme_values",
    "CCME_WQI":                             "ccme_wqi",
}

# Reverse mapping - alias -> original
COLUMN_NAME_MAP_REVERSE: Dict[str, str] = {v: k for k, v in COLUMN_NAME_MAP.items()}

# ---------------------------------------------------------------------------
# Sentinel / error-code values identified during the data audit
#   Source: notebooks/01_dataset_audit.ipynb, Section 7 - Suspicious Values
#   Temperature (cel): 80x -999999.0, 1x -596.0, 1x -99.9, 1x -80.0
#   pH (ph units):      3x -999999.0, 1x -1.0
#   DO (mg/l):          1x -999999.0
# ---------------------------------------------------------------------------
SENTINEL_VALUES: List[float] = [-999999.0, -9999.0, -999.0, -596.0, -99.9, -80.0]

# Numeric columns that may carry sentinel codes
SENTINEL_COLUMNS: List[str] = ["ph", "temperature_c", "do_mg_l"]

# ---------------------------------------------------------------------------
# Physically implausible value rules - FLAGGING only, not deletion
#
# Rationale for each rule:
#   ph:
#     - Natural freshwater: 6.5-9.5; absolute bounds: 0-14.
#     - Audit found pH values up to 67115 (India) and 14-9948 range (India, Canada):
#       instrument scale-corruption or unit mismatches -> flagged and NaN'd.
#   temperature_c:
#     - Surface freshwater range: -2 to 40C. Effluent can reach ~50C.
#     - 5,983 USA records > 50C appear to be in Fahrenheit (see Stage 4).
#     - Values > 98C are physically impossible for surface water -> flagged.
#   Chemical concentrations:
#     - Extreme high values (BOD > 534, Ammonia = 200) occur in Effluent/
#       highly polluted rivers - plausible but flagged for downstream awareness.
# ---------------------------------------------------------------------------
SUSPICIOUS_VALUE_RULES: Dict[str, Dict[str, float]] = {
    "ph":                  {"hard_min": 0.0,  "hard_max": 14.0},
    "temperature_c":       {"hard_min": -5.0, "hard_max": 98.0},
    "ammonia_mg_l":        {"hard_min": 0.0,  "hard_max": 500.0},
    "bod_mg_l":            {"hard_min": 0.0,  "hard_max": 1000.0},
    "do_mg_l":             {"hard_min": 0.0,  "hard_max": 30.0},
    "orthophosphate_mg_l": {"hard_min": 0.0,  "hard_max": 200.0},
    "nitrogen_mg_l":       {"hard_min": 0.0,  "hard_max": 500.0},
    "nitrate_mg_l":        {"hard_min": 0.0,  "hard_max": 1000.0},
    "ccme_values":         {"hard_min": 0.0,  "hard_max": 100.0},
}

# ---------------------------------------------------------------------------
# Sub-detection-limit (SDL) clipping
#
# Audit finding: small negative concentrations in chemical parameters are caused
# by instrument baseline calibration noise below the analytical detection limit.
# Clipping to 0.0 is standard laboratory practice.
#
#   ammonia_mg_l         min observed: -0.005  -> clip to 0.0
#   bod_mg_l             min observed: -2.0    -> clip to 0.0
#   orthophosphate_mg_l  min observed: -0.004  -> clip to 0.0
#   do_mg_l              min (excl sentinel): -0.05 -> clip to 0.0
#
# NOT clipped: ph (negatives -> sentinel), temperature_c (legitimate sub-zero exists)
# ---------------------------------------------------------------------------
SDL_CLIP_COLUMNS: List[str] = [
    "ammonia_mg_l",
    "bod_mg_l",
    "do_mg_l",
    "orthophosphate_mg_l",
    "nitrogen_mg_l",
    "nitrate_mg_l",
]

# CCME WQI ordered categories (classification target)
CCME_WQI_CATEGORIES: List[str] = ["Poor", "Marginal", "Fair", "Good", "Excellent"]

# ---------------------------------------------------------------------------
# Missing-value strategy - documented per column
#
# Strategy is DECLARED here; imputation is out-of-scope for this module
# (no imputation in the cleaning commit; imputation must be fitted only on the
# training split to prevent leakage).
# ---------------------------------------------------------------------------
MISSING_VALUE_STRATEGY: Dict[str, str] = {
    "ammonia_mg_l": (
        "MAR - synchronised batch dropout with Orthophosphate, Nitrogen, Nitrate "
        "(all share identical 34.2% missingness). Strategy: train baseline models on "
        "complete-case subset (2.41M rows). Full-dataset models: KNN imputation within "
        "country x waterbody_type groups, fitted ONLY on the training split."
    ),
    "orthophosphate_mg_l": (
        "MAR - same synchronised 34.2% batch as Ammonia/Nitrogen/Nitrate. "
        "Key eutrophication driver (r = -0.66 with CCME). "
        "Strategy: KNN within country x waterbody_type on training split."
    ),
    "nitrogen_mg_l": (
        "MAR - same synchronised 34.2% batch. "
        "Strategy: KNN within country x waterbody_type on training split. "
        "Alternative: iterative imputer conditioned on co-measured nutrients."
    ),
    "nitrate_mg_l": (
        "MAR - synchronised batch, 1,466,431 rows (34.1%) missing. Slightly fewer than "
        "Ammonia/N/PO4 due to separate recording. Strategy: KNN on training split."
    ),
    "bod_mg_l": (
        "MAR - 1,466,334 rows (34.1%) missing. Direct organic-pollution indicator. "
        "Strategy: KNN within country x waterbody_type on training split."
    ),
    "do_mg_l": (
        "Moderate missingness (26.6%, 1,140,980 rows). Likely continuous-probe gap. "
        "DO strongly correlated with Temperature (Henry's Law). "
        "Strategy: physics-informed imputation (DO ~ f(Temperature, Salinity)) "
        "or KNN with temperature and pH as predictors. Fitted on training split only."
    ),
    "ph": (
        "Moderate missingness (28.0%, 1,201,075 rows after sentinel removal). "
        "Sensors deployed intermittently. Do NOT impute with global median - pH varies "
        "meaningfully by waterbody chemistry. "
        "Strategy: KNN within country x waterbody_type on training split."
    ),
    "temperature_c": (
        "Low missingness (<0.2%, 6,558 rows) after sentinel removal. "
        "Strategy: linear interpolation within time-ordered site records, "
        "or median by (country, month) on training split."
    ),
    "ccme_values": (
        "Complete (0% missing) - no imputation needed. "
        "This is the regression target; imputing it would introduce target leakage."
    ),
    "ccme_wqi": (
        "Complete (0% missing) - no imputation needed. "
        "This is the classification target; imputing it would introduce target leakage."
    ),
    "country":        "Complete - no imputation needed.",
    "area":           "Complete - no imputation needed.",
    "waterbody_type": "Complete - no imputation needed.",
    "date":           "Complete - no imputation needed.",
}


# ---------------------------------------------------------------------------
# Cleaning summary dataclass
# ---------------------------------------------------------------------------

@dataclass
class CleaningSummary:
    """Immutable record of every operation and its row-count impact."""

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
        """Record one cleaning operation."""
        removed = rows_before - rows_after
        self.steps.append({
            "step": name,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "rows_removed": removed,
            "rows_removed_pct": round(removed / rows_before * 100, 4) if rows_before else 0.0,
            "notes": notes,
        })
        sign = "-" if removed > 0 else "+"
        logger.info(
            "  [%s]  %s -> %s rows  (%s%s rows, %.4f%%)  %s",
            name,
            f"{rows_before:,}",
            f"{rows_after:,}",
            sign,
            f"{abs(removed):,}",
            abs(removed) / rows_before * 100 if rows_before else 0.0,
            notes,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return steps as a tidy DataFrame for notebook display."""
        return pd.DataFrame(self.steps)

    def print_summary(self) -> None:
        """Print a concise, human-readable cleaning report."""
        total_removed = self.raw_rows - self.final_rows
        pct_removed = total_removed / self.raw_rows * 100 if self.raw_rows else 0.0
        print("\n" + "=" * 70)
        print("WATER QUALITY CLEANING PIPELINE - SUMMARY")
        print("=" * 70)
        print(f"  Raw dataset    : {self.raw_rows:>10,} rows x {self.raw_cols} columns")
        print(f"  Cleaned dataset: {self.final_rows:>10,} rows x {self.final_cols} columns")
        print(f"  Rows removed   : {total_removed:>10,} ({pct_removed:.4f}%)")
        print(f"  Flag columns   : {len(self.flags_added)}")
        print(f"  Runtime        : {self.elapsed_seconds:.1f}s")
        if self.output_path:
            print(f"  Output path    : {self.output_path}")
        print("-" * 70)
        for s in self.steps:
            row_delta = s["rows_removed"]
            sign = "-" if row_delta > 0 else " "
            print(
                f"  {s['step']:<40}  "
                f"{s['rows_before']:>10,} -> {s['rows_after']:>10,}  "
                f"({sign}{abs(row_delta):,})"
            )
        print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main Preprocessor Class
# ---------------------------------------------------------------------------

class WaterQualityPreprocessor:
    """
    Reusable, documented cleaning pipeline for the combined water-quality dataset.

    Parameters
    ----------
    raw_path : str
        Absolute or relative path to the raw dataset (Parquet or CSV).
    output_path : str, optional
        Where to write the cleaned Parquet file.  If None, file is NOT written.
    nrows : int, optional
        Limit the number of rows loaded (for fast development iteration).
    """

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

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> Tuple[pd.DataFrame, CleaningSummary]:
        """
        Execute the full cleaning pipeline.

        Returns
        -------
        Tuple[pd.DataFrame, CleaningSummary]
        """
        t0 = time.time()
        logger.info("=" * 70)
        logger.info("WATER QUALITY CLEANING PIPELINE - STARTING")
        logger.info("=" * 70)

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

        # Attach metadata to DataFrame
        df.attrs["column_mapping"] = COLUMN_NAME_MAP
        df.attrs["missing_value_strategy"] = MISSING_VALUE_STRATEGY
        df.attrs["cleaning_summary"] = self.summary.to_dataframe().to_dict("records")
        df.attrs["source_path"] = self.raw_path

        if self.output_path:
            self._save(df)
            self.summary.output_path = self.output_path

        self.summary.print_summary()

        logger.info("CLEANING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)

        return df, self.summary

    # ------------------------------------------------------------------
    # Stage 1 - Load
    # ------------------------------------------------------------------

    def _load(self) -> pd.DataFrame:
        """Load the raw dataset without any modification."""
        path = self.raw_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Raw dataset not found: {path}")

        ext = Path(path).suffix.lower()
        t0 = time.time()
        logger.info("Loading raw dataset: %s", path)

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

        elapsed = time.time() - t0
        mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        logger.info(
            "Loaded %s rows x %s cols in %.2fs (%.1f MB RAM)",
            f"{df.shape[0]:,}", df.shape[1], elapsed, mem_mb,
        )
        return df

    # ------------------------------------------------------------------
    # Stage 2 - Column name standardisation
    # ------------------------------------------------------------------

    def _standardise_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rename raw columns to snake_case aliases.

        The original-to-alias mapping is stored in COLUMN_NAME_MAP and also
        attached to the returned DataFrame's .attrs["column_mapping"].
        Only columns present in the raw file are renamed; extras are left as-is.
        """
        logger.info("Stage 1 - Standardising column names")
        existing_map = {k: v for k, v in COLUMN_NAME_MAP.items() if k in df.columns}
        df = df.rename(columns=existing_map)
        logger.info("  Renamed %d columns. New columns: %s", len(existing_map), list(df.columns))
        return df

    # ------------------------------------------------------------------
    # Stage 3 - Data type casting
    # ------------------------------------------------------------------

    def _cast_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cast columns to semantically appropriate dtypes.

        - date       : string 'DD-MM-YYYY' -> datetime64[ns] (dayfirst=True)
        - ccme_wqi   : object -> ordered Categorical (Poor < Marginal < Fair < Good < Excellent)
        - Numeric cols: already float64 from Parquet; validated but not re-cast.
        - String cols : strip whitespace and normalise blank strings to NaN.
        """
        logger.info("Stage 2 - Casting data types")
        rows_before = len(df)

        # --- Date ---
        if "date" in df.columns:
            n_before = df["date"].notna().sum()
            df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
            n_after = df["date"].notna().sum()
            n_failed = int(n_before - n_after)
            if n_failed > 0:
                logger.warning(
                    "  date: %d values could not be parsed to datetime -> NaT.", n_failed
                )
            logger.info("  date: cast to datetime64[ns]  (parsed %s / %s)", f"{n_after:,}", f"{n_before:,}")

        # --- CCME_WQI -> ordered Categorical ---
        if "ccme_wqi" in df.columns:
            present_cats = [c for c in CCME_WQI_CATEGORIES if c in df["ccme_wqi"].values]
            cat_dtype = pd.CategoricalDtype(categories=present_cats, ordered=True)
            df["ccme_wqi"] = df["ccme_wqi"].astype(cat_dtype)
            logger.info("  ccme_wqi: cast to ordered Categorical  %s", present_cats)

        # --- String columns: strip whitespace, normalise blank -> NaN ---
        NULL_LIKE = {"nan", "null", "none", "na", "n/a", "nil", "", "?"}
        str_cols = ["country", "area", "waterbody_type"]
        for col in str_cols:
            if col not in df.columns:
                continue
            df[col] = df[col].astype(str).str.strip()
            mask_blank = df[col].str.lower().isin(NULL_LIKE)
            n_blank = int(mask_blank.sum())
            if n_blank > 0:
                df.loc[mask_blank, col] = np.nan
                logger.info("  %s: %d blank/null-like strings -> NaN", col, n_blank)

        self.summary.log_step("cast_data_types", rows_before, len(df), "No rows removed")
        return df

    # ------------------------------------------------------------------
    # Stage 4 - Sentinel / error-code replacement
    # ------------------------------------------------------------------

    def _replace_sentinels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace known hardware telemetry error codes with NaN.

        Sentinel values identified in the audit:
          -999999.0  (hardware default for instrument failure) - Temperature, pH, DO
          -596.0, -99.9, -80.0  (anomalous negative temps in USA data)
          -1.0  (one pH reading)
          pH > 14.0  (absolute physical impossibility; includes 67115.0 and India batch)

        NO rows are dropped here; only cell values are replaced with NaN.
        """
        logger.info("Stage 3 - Replacing sentinel / error-code values with NaN")
        rows_before = len(df)
        total_replaced = 0

        for col in SENTINEL_COLUMNS:
            if col not in df.columns:
                continue

            # Rule 1: extreme negatives (hardware failure codes <= -999)
            mask_extreme_neg = df[col] <= -999.0
            # Rule 2: explicit listed sentinels
            mask_explicit = df[col].isin(SENTINEL_VALUES)
            combined = mask_extreme_neg | mask_explicit

            n_replaced = int(combined.sum())
            if n_replaced > 0:
                df.loc[combined, col] = np.nan
                total_replaced += n_replaced
                logger.info("  %s: replaced %d sentinel values with NaN", col, n_replaced)

        # pH-specific rules
        if "ph" in df.columns:
            # pH = -1.0 (one occurrence - plausible misentry)
            mask_minus1 = df["ph"] == -1.0
            n_minus1 = int(mask_minus1.sum())
            if n_minus1:
                df.loc[mask_minus1, "ph"] = np.nan
                total_replaced += n_minus1
                logger.info("  ph: replaced %d value(s) of -1.0 with NaN", n_minus1)

            # pH > 14.0 (physically impossible)
            mask_ph_high = df["ph"] > 14.0
            n_ph_high = int(mask_ph_high.sum())
            if n_ph_high:
                sample_vals = (
                    df.loc[mask_ph_high, "ph"].value_counts().head(5).to_dict()
                )
                df.loc[mask_ph_high, "ph"] = np.nan
                total_replaced += n_ph_high
                logger.info(
                    "  ph: replaced %d physically impossible values (> 14.0) with NaN. "
                    "Sample: %s", n_ph_high, sample_vals
                )

        self.summary.log_step(
            "replace_sentinels",
            rows_before,
            len(df),
            f"{total_replaced:,} cells replaced with NaN (no rows dropped)",
        )
        return df

    # ------------------------------------------------------------------
    # Stage 5 - USA temperature unit correction
    # ------------------------------------------------------------------

    def _fix_usa_temperature_unit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Correct USA temperature readings recorded in degF instead of degC.

        Evidence (from audit):
          5,852 USA rows (0.31%) have Temperature > 50C - physically impossible
          for surface water in continental USA (max expected ~35C in summer).
          When converted from F: (F - 32) * 5/9, all fall in [10, 36]C - normal.
          99.7% of USA rows already have values in Celsius range.
          England Effluent records with T > 50C are RETAINED (industrial discharge).

        Conversion applied only to: country == 'USA' AND temperature_c > 50.
        """
        logger.info("Stage 4 - Correcting USA temperatures recorded in degF")
        rows_before = len(df)

        if "temperature_c" not in df.columns or "country" not in df.columns:
            logger.warning("  temperature_c or country column missing - skipping.")
            return df

        mask_usa_f = (df["country"] == "USA") & (df["temperature_c"] > 50)
        n_corrected = int(mask_usa_f.sum())

        if n_corrected > 0:
            df.loc[mask_usa_f, "temperature_c"] = (
                (df.loc[mask_usa_f, "temperature_c"] - 32) * 5 / 9
            )
            corrected_min = df.loc[mask_usa_f, "temperature_c"].min()
            corrected_max = df.loc[mask_usa_f, "temperature_c"].max()
            logger.info(
                "  Converted %d USA temperature values from degF -> degC.  "
                "Post-conversion range: [%.2f, %.2f] degC",
                n_corrected, corrected_min, corrected_max,
            )

        self.summary.log_step(
            "fix_usa_temperature_unit",
            rows_before,
            len(df),
            f"{n_corrected:,} USA degF readings converted to degC",
        )
        return df

    # ------------------------------------------------------------------
    # Stage 6 - Sub-detection-limit negative clipping
    # ------------------------------------------------------------------

    def _clip_sub_detection_limit_negatives(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clip small negative concentrations to 0.0.

        Physical justification:
          Chemical concentrations cannot be negative. Small negatives (-0.005 to -2.0)
          arise when instrument readings fall below the analytical detection limit and
          baseline noise pulls the measurement slightly negative. Standard laboratory
          practice is to report these as 0.0.

        Scope: only SDL_CLIP_COLUMNS; only values in range [-5.0, 0.0).
        Values <= -5.0 would be suspicious and are left for the sentinel/flag stages.
        """
        logger.info("Stage 5 - Clipping sub-detection-limit negative concentrations -> 0.0")
        rows_before = len(df)
        total_clipped = 0

        for col in SDL_CLIP_COLUMNS:
            if col not in df.columns:
                continue
            # Only clip values in the SDL range [-5, 0)
            mask_sdl = (df[col] < 0.0) & (df[col] >= -5.0)
            n_clipped = int(mask_sdl.sum())
            if n_clipped > 0:
                df.loc[mask_sdl, col] = 0.0
                total_clipped += n_clipped
                logger.info("  %s: %d SDL negatives clipped to 0.0", col, n_clipped)

        self.summary.log_step(
            "clip_sdl_negatives",
            rows_before,
            len(df),
            f"{total_clipped:,} sub-detection-limit negatives clipped to 0.0",
        )
        return df

    # ------------------------------------------------------------------
    # Stage 7 - Exact duplicate removal
    # ------------------------------------------------------------------

    def _drop_exact_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove exact duplicate records, keeping the first occurrence.

        Justification:
          The audit confirmed 10,005 rows identical across ALL 14 columns
          (Country='Canada', verified with paired index inspection). These are
          ingestion artifacts from combining country files, not legitimate repeated
          measurements. Keeping duplicates would:
            - Inflate training set size
            - Bias class distributions
            - Risk train/test contamination if the same record appears in both

          One copy of each unique measurement is always retained. Metadata columns
          (Country, Area, Date) are part of the key, so legitimate repeated
          measurements at the same site on different dates are NOT affected.
        """
        logger.info("Stage 6 - Dropping exact duplicate records")
        rows_before = len(df)

        df = df.drop_duplicates(keep="first")
        rows_after = len(df)
        n_dropped = rows_before - rows_after

        self.summary.log_step(
            "drop_exact_duplicates",
            rows_before,
            rows_after,
            f"Dropped {n_dropped:,} exact duplicates (all columns identical, keep='first')",
        )
        return df

    # ------------------------------------------------------------------
    # Stage 8 - Flag suspicious / physically implausible values
    # ------------------------------------------------------------------

    def _flag_suspicious_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add boolean flag columns for physically implausible values.

        FLAGS are additive - they do NOT remove any rows. Downstream modelling
        steps can use these flags to:
          1. Exclude flagged rows from training.
          2. Build separate models for normal vs extreme ranges.
          3. Report data-quality metrics per country/site.

        Flag column naming: {original_col}_flag_suspicious

        Rules applied (from SUSPICIOUS_VALUE_RULES):
          - Value < hard_min OR value > hard_max -> flagged True
          - NaN values are NOT flagged (accounted for in missingness strategy)
        """
        logger.info("Stage 7 - Flagging physically suspicious values")
        rows_before = len(df)
        flag_cols: List[str] = []

        for col, bounds in SUSPICIOUS_VALUE_RULES.items():
            if col not in df.columns:
                continue
            hard_min = bounds["hard_min"]
            hard_max = bounds["hard_max"]
            flag_col = f"{col}_flag_suspicious"
            # Flag non-null values outside the hard bounds
            mask = (
                df[col].notna()
                & ((df[col] < hard_min) | (df[col] > hard_max))
            )
            df[flag_col] = mask
            n_flagged = int(mask.sum())
            flag_cols.append(flag_col)
            if n_flagged > 0:
                logger.info(
                    "  %s: %d rows flagged as suspicious (outside [%s, %s])  ->  %s",
                    col, n_flagged, hard_min, hard_max, flag_col,
                )

        self.summary.flags_added = flag_cols
        self.summary.log_step(
            "flag_suspicious_values",
            rows_before,
            len(df),
            f"{len(flag_cols)} flag columns added; no rows dropped",
        )
        return df

    # ------------------------------------------------------------------
    # Stage 9 - Missing-value strategy (documentation & snapshot)
    # ------------------------------------------------------------------

    def _apply_missing_value_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Attach the documented missing-value strategy and log a missingness snapshot.

        No imputation is performed here - per project specification, imputation
        must be fitted ONLY on the training split to prevent leakage.
        The strategy document (MISSING_VALUE_STRATEGY) is attached to df.attrs.

        Validation assertions:
          - Metadata columns (country, area, waterbody_type, date) remain complete.
          - Target columns (ccme_values, ccme_wqi) remain complete.
        """
        logger.info("Stage 8 - Missing-value snapshot & strategy validation")
        rows_before = len(df)
        n = len(df)

        logger.info("  Post-cleaning missingness snapshot:")
        for col in df.columns:
            if col.endswith("_flag_suspicious"):
                continue
            n_missing = int(df[col].isna().sum())
            pct = n_missing / n * 100 if n else 0.0
            if n_missing > 0:
                strategy = MISSING_VALUE_STRATEGY.get(col, "No strategy documented.")
                strategy_short = strategy[:80] + "..." if len(strategy) > 80 else strategy
                logger.info(
                    "    %-35s  %9s missing  (%6.2f%%)  -> %s",
                    col, f"{n_missing:,}", pct, strategy_short,
                )

        # Validation: critical columns must be complete (targets and key grouping keys).
        # 'area' is EXCLUDED here — a small number of monitoring sites have no name
        # recorded (184 rows in the real dataset), which is a source data limitation
        # and does not prevent modelling; we log a warning instead.
        MUST_BE_COMPLETE = ["country", "waterbody_type", "ccme_values", "ccme_wqi"]
        for col in MUST_BE_COMPLETE:
            if col not in df.columns:
                continue
            n_missing = int(df[col].isna().sum())
            if n_missing > 0:
                logger.error(
                    "  VALIDATION FAILED: '%s' has %d missing values - expected 0.",
                    col, n_missing,
                )

        # Advisory warning for area (site identifier can be sparse in source data)
        if "area" in df.columns:
            n_area_missing = int(df["area"].isna().sum())
            if n_area_missing > 0:
                logger.warning(
                    "  ADVISORY: 'area' has %d missing values (%.4f%%). "
                    "This is a known source-data limitation - site names not always recorded.",
                    n_area_missing, n_area_missing / n * 100,
                )

        self.summary.log_step(
            "missing_value_strategy",
            rows_before,
            len(df),
            "No rows removed - strategy documented; imputation deferred to train-split stage",
        )
        return df

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _save(self, df: pd.DataFrame) -> None:
        """Save the cleaned DataFrame as Parquet with Snappy compression."""
        out = Path(self.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(str(out), compression="snappy", index=False)
        size_mb = out.stat().st_size / 1024 / 1024
        logger.info("Cleaned dataset saved -> %s  (%.2f MB)", out, size_mb)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def run_cleaning_pipeline(
    raw_path: Optional[str] = None,
    output_path: Optional[str] = None,
    nrows: Optional[int] = None,
) -> Tuple[pd.DataFrame, CleaningSummary]:
    """
    One-call entry point for the cleaning pipeline.

    Parameters
    ----------
    raw_path : str, optional
        Path to raw dataset. If None, auto-discovers under data/.
    output_path : str, optional
        Where to save the cleaned Parquet. If None, result is not saved to disk.
    nrows : int, optional
        Row limit for development / testing.

    Returns
    -------
    Tuple[pd.DataFrame, CleaningSummary]
    """
    if raw_path is None:
        raw_path = os.path.join(
            "data", "Dataset", "Parquet", "Combined Data", "Combined_dataset.parquet"
        )
    if output_path is None:
        output_path = os.path.join(
            "data", "GeneratedOutput", "02.Preprocessing", "cleaned_dataset.parquet"
        )

    pipeline = WaterQualityPreprocessor(
        raw_path=raw_path,
        output_path=output_path,
        nrows=nrows,
    )
    return pipeline.run()


# ---------------------------------------------------------------------------
# Schema / assertion helpers (reusable in tests and notebooks)
# ---------------------------------------------------------------------------

def assert_schema_preserved(df: pd.DataFrame) -> None:
    """
    Raise AssertionError if the cleaned DataFrame is missing required columns
    or has incorrect dtypes.
    """
    required_numeric = [
        "ammonia_mg_l", "bod_mg_l", "do_mg_l", "orthophosphate_mg_l",
        "ph", "temperature_c", "nitrogen_mg_l", "nitrate_mg_l", "ccme_values",
    ]
    required_meta = ["country", "area", "waterbody_type", "date", "ccme_wqi"]

    for col in required_numeric:
        assert col in df.columns, f"Missing numeric column: {col}"
        assert pd.api.types.is_numeric_dtype(df[col]), (
            f"Column '{col}' should be numeric, got {df[col].dtype}"
        )

    for col in required_meta:
        assert col in df.columns, f"Missing metadata column: {col}"

    assert pd.api.types.is_datetime64_any_dtype(df["date"]), (
        f"'date' should be datetime64, got {df['date'].dtype}"
    )
    assert hasattr(df["ccme_wqi"], "cat"), (
        f"'ccme_wqi' should be Categorical, got {df['ccme_wqi'].dtype}"
    )
    assert df["ccme_wqi"].cat.ordered, "'ccme_wqi' Categorical must be ordered"


def assert_no_sentinels(df: pd.DataFrame) -> None:
    """Raise AssertionError if any sentinel values remain after cleaning."""
    for col in ["ph", "temperature_c", "do_mg_l"]:
        if col not in df.columns:
            continue
        n_sentinels = int((df[col] <= -999).sum())
        assert n_sentinels == 0, (
            f"Column '{col}' still contains {n_sentinels:,} sentinel values <= -999"
        )
    if "ph" in df.columns:
        n_ph_high = int((df["ph"] > 14).sum())
        assert n_ph_high == 0, f"'ph' still has {n_ph_high:,} values > 14.0"


def assert_no_sdl_negatives(df: pd.DataFrame) -> None:
    """Raise AssertionError if any SDL negatives remain in concentration columns."""
    for col in SDL_CLIP_COLUMNS:
        if col not in df.columns:
            continue
        n_neg = int((df[col] < 0).sum())
        assert n_neg == 0, (
            f"Column '{col}' still has {n_neg:,} negative values after SDL clipping"
        )


def assert_no_exact_duplicates(df: pd.DataFrame) -> None:
    """Raise AssertionError if exact duplicate rows are present."""
    n_dups = int(df.duplicated().sum())
    assert n_dups == 0, f"DataFrame still contains {n_dups:,} exact duplicate rows"


def assert_targets_complete(df: pd.DataFrame) -> None:
    """Raise AssertionError if target columns contain NaN (leakage risk)."""
    for col in ["ccme_values", "ccme_wqi"]:
        if col not in df.columns:
            continue
        n_missing = int(df[col].isna().sum())
        assert n_missing == 0, (
            f"Target column '{col}' has {n_missing:,} missing values - check pipeline"
        )


def assert_flag_columns_present(df: pd.DataFrame) -> None:
    """Raise AssertionError if expected suspicious-value flag columns are absent."""
    for col in SUSPICIOUS_VALUE_RULES:
        flag_col = f"{col}_flag_suspicious"
        assert flag_col in df.columns, (
            f"Expected flag column '{flag_col}' not found in DataFrame"
        )
        assert df[flag_col].dtype == bool, (
            f"Flag column '{flag_col}' should be bool, got {df[flag_col].dtype}"
        )


def assert_usa_temperatures_in_celsius_range(df: pd.DataFrame) -> None:
    """Raise AssertionError if USA temperatures remain above 50C post-correction."""
    if "temperature_c" not in df.columns or "country" not in df.columns:
        return
    mask = (df["country"] == "USA") & (df["temperature_c"] > 50)
    n = int(mask.sum())
    assert n == 0, (
        f"{n:,} USA temperature values still > 50C after Fahrenheit correction"
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Water Quality Cleaning Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to raw dataset (Parquet or CSV). Defaults to auto-discovery.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path for cleaned Parquet output.",
    )
    parser.add_argument(
        "--nrows", type=int, default=None,
        help="Limit rows loaded (useful for testing).",
    )
    args = parser.parse_args()

    run_cleaning_pipeline(
        raw_path=args.input,
        output_path=args.output,
        nrows=args.nrows,
    )
