"""Unit tests for water quality data preprocessing, feature engineering, and preparation."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import (
    BASE_NUMERIC_FEATURES,
    CCME_WQI_CATEGORIES,
    CCME_WQI_MAP,
    COLUMN_NAME_MAP,
    ENGINEERED_FEATURES,
    MISSING_VALUE_STRATEGY,
    MODEL_INPUT_FEATURES,
    NEGATIVE_CLEANING_RULES,
    SDL_CLIP_COLUMNS,
    SENTINEL_COLUMNS,
    SENTINEL_VALUES,
    SUSPICIOUS_VALUE_RULES,
    TARGET_CLASSIFICATION,
    TARGET_CLASSIFICATION_ORDINAL,
    TARGET_REGRESSION,
    CleaningSummary,
    WaterQualityPreprocessor,
    assert_flag_columns_present,
    assert_no_exact_duplicates,
    assert_no_sdl_negatives,
    assert_no_sentinels,
    assert_raw_unmodified,
    assert_schema_preserved,
    assert_targets_complete,
    assert_usa_temperatures_in_celsius_range,
    calculate_do_saturation,
    engineer_features,
    generate_data_quality_summary,
    impute_missing_values,
    prepare_training_datasets,
    run_cleaning_pipeline,
    scale_features,
    split_data,
    load_training_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_raw_fixture(n_extra_duplicates: int = 2) -> pd.DataFrame:
    """
    Build a minimal DataFrame that mirrors the raw combined dataset schema
    (original column names, original dtypes) with intentional quality issues:
      - Sentinel temperature: -999999.0
      - pH sentinels: -999999.0, -1.0, 67115.0, 15.0
      - USA temperature > 50 (Fahrenheit)
      - Negative noise in Ammonia (-0.003 mg/L) and DO (-0.05 mg/L)
      - Unphysical negative BOD (-1.5 mg/L)
      - Exact duplicate rows
    """
    data = {
        "Country":                              ["England", "USA",  "USA",   "India", "Canada", "Australia", "Ireland"],
        "Area":                                 ["River A", "Lake B", "Lake B", "River C", "Site D", "Site E", "Site F"],
        "Waterbody Type":                       ["River", "Lake", "Lake", "River", "River", "River", "Stream"],
        "Date":                                 ["15-03-2010", "01-07-2015", "01-07-2015", "10-06-2018", "20-01-2000", "05-09-2005", "12-11-2019"],
        "Ammonia (mg/l)":                       [0.05,   0.10,   0.10,   0.30,   0.02,   -0.003, 0.07],
        "Biochemical Oxygen Demand (mg/l)":     [2.0,    4.0,    4.0,   -1.5,   1.0,    0.5,    3.2],
        "Dissolved Oxygen (mg/l)":              [8.5,    7.0,    7.0,    5.5,    9.0,   -0.05,  8.0],
        "Orthophosphate (mg/l)":               [0.01,   0.02,   0.02,   0.05,   0.01,  -0.002, 0.03],
        "pH (ph units)":                        [7.5,   -1.0,   -1.0,  67115.0, 8.0,   15.0,   7.2],
        "Temperature (cel)":                    [12.0,  68.0,   68.0,  25.0,    8.0,   22.0,   15.0],
        "Nitrogen (mg/l)":                      [1.5,    2.0,    2.0,   3.0,    0.8,    1.2,    1.0],
        "Nitrate (mg/l)":                       [5.0,    8.0,    8.0,   6.0,    4.0,    3.0,    7.0],
        "CCME_Values":                          [95.0,  88.0,   88.0,  72.0,   100.0,  91.0,   85.0],
        "CCME_WQI":                             ["Excellent", "Good", "Good", "Fair", "Excellent", "Good", "Good"],
    }

    df = pd.DataFrame(data)
    df.loc[0, "Temperature (cel)"] = -999999.0
    df.loc[1, "Dissolved Oxygen (mg/l)"] = -999999.0

    # Add duplicate rows of row 4
    dup_rows = pd.concat([df.iloc[[4]]] * n_extra_duplicates, ignore_index=True)
    df = pd.concat([df, dup_rows], ignore_index=True)
    return df


@pytest.fixture
def raw_fixture() -> pd.DataFrame:
    return _make_raw_fixture(n_extra_duplicates=2)


@pytest.fixture
def cleaned_fixture(raw_fixture, tmp_path) -> tuple:
    raw_path = tmp_path / "raw_fixture.parquet"
    raw_fixture.to_parquet(str(raw_path), compression="snappy", index=False)
    pipeline = WaterQualityPreprocessor(raw_path=str(raw_path), output_path=None)
    return pipeline.run()


# ---------------------------------------------------------------------------
# 1. Column name standardisation
# ---------------------------------------------------------------------------

class TestColumnNameStandardisation:

    def test_all_original_columns_renamed(self, cleaned_fixture):
        df, _ = cleaned_fixture
        for original_col in COLUMN_NAME_MAP.keys():
            assert original_col not in df.columns

    def test_all_alias_columns_present(self, cleaned_fixture):
        df, _ = cleaned_fixture
        for alias in COLUMN_NAME_MAP.values():
            assert alias in df.columns

    def test_original_column_mapping_in_attrs(self, cleaned_fixture):
        df, _ = cleaned_fixture
        assert "column_mapping" in df.attrs
        assert df.attrs["column_mapping"] == COLUMN_NAME_MAP


# ---------------------------------------------------------------------------
# 2. Data type casting
# ---------------------------------------------------------------------------

class TestDataTypeCasting:

    def test_date_cast_to_datetime(self, cleaned_fixture):
        df, _ = cleaned_fixture
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_date_parsed_correctly(self, cleaned_fixture):
        df, _ = cleaned_fixture
        assert df["date"].dropna().iloc[0].year == 2010

    def test_ccme_wqi_is_ordered_categorical(self, cleaned_fixture):
        df, _ = cleaned_fixture
        assert hasattr(df["ccme_wqi"], "cat")
        assert df["ccme_wqi"].cat.ordered

    def test_ccme_wqi_category_order(self, cleaned_fixture):
        df, _ = cleaned_fixture
        cats = list(df["ccme_wqi"].cat.categories)
        expected = [c for c in CCME_WQI_CATEGORIES if c in cats]
        assert cats == expected

    def test_numeric_columns_are_float(self, cleaned_fixture):
        df, _ = cleaned_fixture
        numeric_cols = [
            "ammonia_mg_l", "bod_mg_l", "do_mg_l", "orthophosphate_mg_l",
            "ph", "temperature_c", "nitrogen_mg_l", "nitrate_mg_l", "ccme_values",
        ]
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(df[col])


# ---------------------------------------------------------------------------
# 3. Sentinel and physical bounds replacement
# ---------------------------------------------------------------------------

class TestSentinelReplacement:

    def test_sentinel_temperature_replaced_with_nan(self, cleaned_fixture):
        df, _ = cleaned_fixture
        assert pd.isna(df["temperature_c"].iloc[0])

    def test_sentinel_do_replaced_with_nan(self, cleaned_fixture):
        df, _ = cleaned_fixture
        assert int((df["do_mg_l"] <= -999.0).sum()) == 0

    def test_ph_minus1_replaced_with_nan(self, cleaned_fixture):
        df, _ = cleaned_fixture
        assert int((df["ph"] < 0.0).sum()) == 0

    def test_ph_scale_corruption_replaced(self, cleaned_fixture):
        df, _ = cleaned_fixture
        assert int((df["ph"] == 67115.0).sum()) == 0

    def test_ph_above_14_replaced_with_nan(self, cleaned_fixture):
        df, _ = cleaned_fixture
        assert int((df["ph"] > 14.0).sum()) == 0

    def test_extreme_negative_temperatures_replaced(self, tmp_path):
        raw = _make_raw_fixture()
        raw.loc[0, "Temperature (cel)"] = -80.0
        raw_path = tmp_path / "extreme_neg_temp.parquet"
        raw.to_parquet(str(raw_path), compression="snappy", index=False)
        pipeline = WaterQualityPreprocessor(raw_path=str(raw_path))
        df_out, _ = pipeline.run()
        assert pd.isna(df_out["temperature_c"].iloc[0])


# ---------------------------------------------------------------------------
# 4. USA temperature correction
# ---------------------------------------------------------------------------

class TestUSATemperatureCorrection:

    def test_usa_temperatures_converted(self, cleaned_fixture):
        df, _ = cleaned_fixture
        usa_rows = df[df["country"] == "USA"]
        assert all(usa_rows["temperature_c"] <= 50.0)

    def test_conversion_value_correct(self, cleaned_fixture):
        df, _ = cleaned_fixture
        usa_rows = df[df["country"] == "USA"]
        expected_c = (68.0 - 32.0) * 5.0 / 9.0
        actual_temps = usa_rows["temperature_c"].dropna().values
        assert all(abs(t - expected_c) < 0.01 for t in actual_temps)

    def test_non_usa_high_temp_not_converted(self, raw_fixture, tmp_path):
        extra = pd.DataFrame({
            "Country": ["England"],
            "Area": ["Industrial Site"],
            "Waterbody Type": ["Effluent"],
            "Date": ["01-01-2020"],
            "Ammonia (mg/l)": [0.5],
            "Biochemical Oxygen Demand (mg/l)": [10.0],
            "Dissolved Oxygen (mg/l)": [3.0],
            "Orthophosphate (mg/l)": [0.1],
            "pH (ph units)": [7.0],
            "Temperature (cel)": [60.0],
            "Nitrogen (mg/l)": [2.0],
            "Nitrate (mg/l)": [5.0],
            "CCME_Values": [55.0],
            "CCME_WQI": ["Marginal"],
        })
        df_combined = pd.concat([raw_fixture, extra], ignore_index=True)
        raw_path = tmp_path / "eng_fixture.parquet"
        df_combined.to_parquet(str(raw_path), compression="snappy", index=False)

        pipeline = WaterQualityPreprocessor(raw_path=str(raw_path))
        df_out, _ = pipeline.run()

        eng_eff = df_out[(df_out["country"] == "England") & (df_out["waterbody_type"] == "Effluent")]
        assert any(eng_eff["temperature_c"] == 60.0)


# ---------------------------------------------------------------------------
# 5. Negative concentration cleaning rules
# ---------------------------------------------------------------------------

class TestNegativeConcentrations:

    def test_detection_limit_noise_clipped_to_zero(self, cleaned_fixture):
        df, _ = cleaned_fixture
        # Ammonia had -0.003 mg/L -> clipped to 0.0
        assert int((df["ammonia_mg_l"] < 0.0).sum()) == 0
        # DO had -0.05 mg/L -> clipped to 0.0
        assert int((df["do_mg_l"] < 0.0).sum()) == 0
        # Orthophosphate had -0.002 mg/L -> clipped to 0.0
        assert int((df["orthophosphate_mg_l"] < 0.0).sum()) == 0

    def test_unphysical_negative_bod_converted_to_nan(self, cleaned_fixture):
        df, _ = cleaned_fixture
        # Row 3 had BOD = -1.5 mg/L (physically invalid consumption) -> converted to NaN
        assert pd.isna(df.loc[df["area"] == "River C", "bod_mg_l"].iloc[0])

    def test_legitimate_sub_zero_temperature_preserved(self, tmp_path):
        raw = _make_raw_fixture()
        raw.loc[0, "Temperature (cel)"] = -2.5
        raw_path = tmp_path / "cold_temp.parquet"
        raw.to_parquet(str(raw_path), compression="snappy", index=False)

        pipeline = WaterQualityPreprocessor(raw_path=str(raw_path))
        df_out, _ = pipeline.run()
        assert any(df_out["temperature_c"] == -2.5)


# ---------------------------------------------------------------------------
# 6. Exact duplicate removal
# ---------------------------------------------------------------------------

class TestDuplicateRemoval:

    def test_no_exact_duplicates_remain(self, cleaned_fixture):
        df, _ = cleaned_fixture
        assert_no_exact_duplicates(df)

    def test_correct_number_of_duplicates_removed(self, raw_fixture, tmp_path):
        raw_path = tmp_path / "dup_fixture.parquet"
        raw_fixture.to_parquet(str(raw_path), compression="snappy", index=False)
        n_exact_dups = int(raw_fixture.duplicated().sum())

        pipeline = WaterQualityPreprocessor(raw_path=str(raw_path))
        _, summary = pipeline.run()

        dedup_step = next(s for s in summary.steps if s["step"] == "drop_exact_duplicates")
        assert dedup_step["rows_removed"] == n_exact_dups

    def test_non_duplicate_rows_retained(self, cleaned_fixture, raw_fixture):
        df, _ = cleaned_fixture
        n_unique_raw = len(raw_fixture.drop_duplicates())
        assert len(df) == n_unique_raw


# ---------------------------------------------------------------------------
# 7. Suspicious value flagging
# ---------------------------------------------------------------------------

class TestSuspiciousValueFlagging:

    def test_all_flag_columns_created(self, cleaned_fixture):
        df, _ = cleaned_fixture
        assert_flag_columns_present(df)

    def test_flag_columns_are_bool(self, cleaned_fixture):
        df, _ = cleaned_fixture
        for col in SUSPICIOUS_VALUE_RULES:
            flag_col = f"{col}_flag_suspicious"
            assert df[flag_col].dtype == bool

    def test_nan_values_not_flagged(self, cleaned_fixture):
        df, _ = cleaned_fixture
        for col in SUSPICIOUS_VALUE_RULES:
            flag_col = f"{col}_flag_suspicious"
            nan_mask = df[col].isna()
            if nan_mask.any():
                assert not df.loc[nan_mask, flag_col].any()


# ---------------------------------------------------------------------------
# 8. Target completeness and mapping
# ---------------------------------------------------------------------------

class TestTargetCompletenessAndMapping:

    def test_ccme_values_no_missing(self, cleaned_fixture):
        df, _ = cleaned_fixture
        assert df["ccme_values"].isna().sum() == 0

    def test_ccme_wqi_no_missing(self, cleaned_fixture):
        df, _ = cleaned_fixture
        assert df["ccme_wqi"].isna().sum() == 0

    def test_ordinal_class_mapping_order(self):
        assert CCME_WQI_MAP["Poor"] == 0
        assert CCME_WQI_MAP["Marginal"] == 1
        assert CCME_WQI_MAP["Fair"] == 2
        assert CCME_WQI_MAP["Good"] == 3
        assert CCME_WQI_MAP["Excellent"] == 4

    def test_targets_strictly_excluded_from_model_inputs(self):
        targets = {TARGET_REGRESSION, TARGET_CLASSIFICATION, TARGET_CLASSIFICATION_ORDINAL}
        for t in targets:
            assert t not in MODEL_INPUT_FEATURES

    def test_model_input_features_authoritative_15_list(self):
        expected_15 = [
            "ph", "temperature_c", "do_mg_l", "bod_mg_l", "ammonia_mg_l",
            "orthophosphate_mg_l", "nitrogen_mg_l", "nitrate_mg_l",
            "np_ratio", "bod_do_ratio", "do_deficit", "month", "quarter",
            "month_sin", "month_cos",
        ]
        assert len(MODEL_INPUT_FEATURES) == 15
        assert MODEL_INPUT_FEATURES == expected_15
        for t in [TARGET_REGRESSION, TARGET_CLASSIFICATION, TARGET_CLASSIFICATION_ORDINAL]:
            assert t not in MODEL_INPUT_FEATURES


# ---------------------------------------------------------------------------
# 9. Cyclical month encoding
# ---------------------------------------------------------------------------

class TestCyclicalMonthEncoding:

    def test_january_and_december_adjacency(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-15", "2020-12-15", "2020-02-15"]),
            "temperature_c": [10.0, 5.0, 8.0],
            "do_mg_l": [10.0, 11.0, 10.5],
            "ph": [7.5, 7.6, 7.4],
            "nitrogen_mg_l": [2.0, 2.5, 2.1],
            "orthophosphate_mg_l": [0.05, 0.04, 0.05],
            "bod_mg_l": [1.5, 1.2, 1.4],
            "ccme_wqi": ["Good", "Good", "Good"],
        })
        out, _ = engineer_features(df)

        # Month 1 (Jan): angle = 0 -> sin = 0, cos = 1
        jan_sin = out.loc[0, "month_sin"]
        jan_cos = out.loc[0, "month_cos"]
        assert np.isclose(jan_sin, 0.0, atol=1e-6)
        assert np.isclose(jan_cos, 1.0, atol=1e-6)

        # Month 12 (Dec): angle = 2*pi*11/12 -> sin = -0.5, cos = sqrt(3)/2
        dec_sin = out.loc[1, "month_sin"]
        dec_cos = out.loc[1, "month_cos"]
        assert np.isclose(dec_sin, -0.5, atol=1e-6)
        assert np.isclose(dec_cos, np.sqrt(3) / 2, atol=1e-6)

        # Distance between Jan and Dec in 2D circular space
        dist_jan_dec = np.sqrt((jan_sin - dec_sin) ** 2 + (jan_cos - dec_cos) ** 2)

        # Month 2 (Feb): angle = 2*pi*1/12 -> sin = 0.5, cos = sqrt(3)/2
        feb_sin = out.loc[2, "month_sin"]
        feb_cos = out.loc[2, "month_cos"]
        dist_jan_feb = np.sqrt((jan_sin - feb_sin) ** 2 + (jan_cos - feb_cos) ** 2)

        # Distances must be equal (exact 30 degree step)
        assert np.isclose(dist_jan_dec, dist_jan_feb, atol=1e-6)

    def test_missing_date_produces_nan_month_encodings(self):
        df = pd.DataFrame({
            "date": [pd.NaT],
            "temperature_c": [15.0],
            "do_mg_l": [8.0],
        })
        out, _ = engineer_features(df)
        assert pd.isna(out.loc[0, "month_sin"])
        assert pd.isna(out.loc[0, "month_cos"])


# ---------------------------------------------------------------------------
# 10. Feature engineering safety and ratios
# ---------------------------------------------------------------------------

class TestFeatureEngineeringSafety:

    def test_np_ratio_division_by_zero(self):
        df = pd.DataFrame({
            "nitrogen_mg_l": [2.0, 3.0, np.nan],
            "orthophosphate_mg_l": [0.0, 0.05, 0.05],
        })
        out, _ = engineer_features(df)
        # Denominator zero must yield NaN, not huge artificial number
        assert pd.isna(out.loc[0, "np_ratio"])
        # Valid denominator yields accurate ratio
        assert np.isclose(out.loc[1, "np_ratio"], 60.0)
        # Missing numerator yields NaN
        assert pd.isna(out.loc[2, "np_ratio"])

    def test_bod_do_ratio_division_by_zero(self):
        df = pd.DataFrame({
            "bod_mg_l": [4.0, 2.0, 1.0],
            "do_mg_l": [0.0, 5.0, np.nan],
        })
        out, _ = engineer_features(df)
        assert pd.isna(out.loc[0, "bod_do_ratio"])
        assert np.isclose(out.loc[1, "bod_do_ratio"], 0.4)
        assert pd.isna(out.loc[2, "bod_do_ratio"])

    def test_signed_dissolved_oxygen_deficit(self):
        # Temp 20C -> saturation ~9.09 mg/L
        sat_20 = calculate_do_saturation(pd.Series([20.0])).iloc[0]
        df = pd.DataFrame({
            "temperature_c": [20.0, 20.0, np.nan],
            "do_mg_l": [5.0, 12.0, 8.0],
        })
        out, _ = engineer_features(df)
        # Normal deficit (oxygen depleted by pollution)
        assert out.loc[0, "do_deficit"] > 0
        assert np.isclose(out.loc[0, "do_deficit"], sat_20 - 5.0)

        # Supersaturation (intense algal photosynthesis) -> signed negative deficit
        assert out.loc[1, "do_deficit"] < 0
        assert np.isclose(out.loc[1, "do_deficit"], sat_20 - 12.0)

        # Missing input yields NaN
        assert pd.isna(out.loc[2, "do_deficit"])

    def test_calculate_do_saturation_sub_zero_and_range(self):
        # Sub-zero temperatures (-5C, -1C) must be bounded at freezing point saturation
        temps = pd.Series([-5.0, -1.0, 0.0, 20.0, 45.0, 55.0, np.nan])
        sat_eh = calculate_do_saturation(temps, formula="elmore_hayes")
        sat_bk = calculate_do_saturation(temps, formula="benson_krause")

        # Elmore-Hayes freezing point saturation is 14.652 mg/L
        assert np.isclose(sat_eh.iloc[0], 14.652)
        assert np.isclose(sat_eh.iloc[1], 14.652)
        assert np.isclose(sat_eh.iloc[2], 14.652)
        assert np.isclose(sat_eh.iloc[3], 9.0218, atol=1e-3)

        # Benson-Krause freezing point saturation is 14.6208 mg/L
        assert np.isclose(sat_bk.iloc[0], 14.6208, atol=1e-3)
        assert np.isclose(sat_bk.iloc[1], 14.6208, atol=1e-3)
        assert np.isclose(sat_bk.iloc[2], 14.6208, atol=1e-3)
        assert np.isclose(sat_bk.iloc[3], 9.0924, atol=1e-3)

        # High temperatures capped at 45.0C
        assert np.isclose(sat_eh.iloc[4], sat_eh.iloc[5])
        assert np.isclose(sat_bk.iloc[4], sat_bk.iloc[5])

        # Missing input returns NaN
        assert pd.isna(sat_eh.iloc[6])
        assert pd.isna(sat_bk.iloc[6])


# ---------------------------------------------------------------------------
# 11. Splitting strategies (Random, Group, Temporal)
# ---------------------------------------------------------------------------

class TestSplittingStrategies:

    def test_stratified_random_split(self, cleaned_fixture):
        df_clean, _ = cleaned_fixture
        df_large = pd.concat([df_clean] * 10, ignore_index=True)
        train_df, val_df, test_df = split_data(
            df_large, train_size=0.70, val_size=0.15, test_size=0.15, method="stratified_random"
        )
        assert len(train_df) + len(val_df) + len(test_df) == len(df_large)
        assert abs(len(train_df) / len(df_large) - 0.70) < 0.05

    def test_country_area_group_split_disjointness(self):
        # Create dataset with multiple country-area clusters
        rows = []
        for station_num in range(20):
            for _ in range(5):
                rows.append({
                    "country": "England",
                    "area": f"Area_{station_num}",
                    "waterbody_type": "River",
                    "ccme_wqi": "Good",
                    "ccme_values": 85.0,
                    "ph": 7.5,
                })
        df_areas = pd.DataFrame(rows)
        # Test both authoritative name and legacy alias
        for method_name in ["country_area_group", "spatial_group"]:
            train_df, val_df, test_df = split_data(df_areas, method=method_name, random_state=42)

            train_areas = set(train_df["area"])
            val_areas = set(val_df["area"])
            test_areas = set(test_df["area"])

            # Country-area clusters across partitions must be 100% mutually exclusive
            assert len(train_areas & val_areas) == 0
            assert len(train_areas & test_areas) == 0
            assert len(val_areas & test_areas) == 0

    def test_temporal_holdout_split(self):
        dates = pd.to_datetime(["2015-06-01", "2019-01-01", "2020-12-31", "2021-05-15", "2022-03-01", "2023-08-20"])
        df_time = pd.DataFrame({
            "date": dates,
            "ccme_wqi": ["Good"] * 6,
            "ph": [7.5] * 6,
        })
        train_df, val_df, test_df = split_data(df_time, method="temporal_holdout")

        assert (train_df["date"].dt.year <= 2020).all()
        assert (val_df["date"].dt.year == 2021).all()
        assert (test_df["date"].dt.year >= 2022).all()

    def test_split_reproducibility(self, cleaned_fixture):
        df_clean, _ = cleaned_fixture
        df_large = pd.concat([df_clean] * 5, ignore_index=True)
        t1, v1, s1 = split_data(df_large, method="stratified_random", random_state=42)
        t2, v2, s2 = split_data(df_large, method="stratified_random", random_state=42)
        pd.testing.assert_frame_equal(t1, t2)
        pd.testing.assert_frame_equal(v1, v2)
        pd.testing.assert_frame_equal(s1, s2)


# ---------------------------------------------------------------------------
# 12. Training-only imputation and scaling
# ---------------------------------------------------------------------------

class TestImputationAndScaling:

    def test_imputation_fitted_on_train_only(self):
        train_data = pd.DataFrame({
            "country": ["USA", "USA"],
            "waterbody_type": ["River", "River"],
            "month": [6, 6],
            "ph": [7.0, 8.0],  # median = 7.5
            "temperature_c": [20.0, 22.0],  # median = 21.0
        })
        val_data = pd.DataFrame({
            "country": ["USA"],
            "waterbody_type": ["River"],
            "month": [6],
            "ph": [np.nan],
            "temperature_c": [np.nan],
        })
        test_data = pd.DataFrame({
            "country": ["USA"],
            "waterbody_type": ["River"],
            "month": [6],
            "ph": [np.nan],
            "temperature_c": [np.nan],
        })

        features = ["ph", "temperature_c"]
        train_imp, val_imp, test_imp, info = impute_missing_values(train_data, val_data, test_data, features)

        assert val_imp.loc[0, "ph"] == 7.5
        assert val_imp.loc[0, "temperature_c"] == 21.0
        assert test_imp.loc[0, "ph"] == 7.5
        assert test_imp.loc[0, "temperature_c"] == 21.0

    def test_unseen_group_fallback_to_global_median(self):
        train_data = pd.DataFrame({
            "country": ["England"],
            "waterbody_type": ["River"],
            "month": [5],
            "ph": [7.4],
        })
        val_data = pd.DataFrame({
            "country": ["Antarctica"],  # unseen group
            "waterbody_type": ["Glacier"],
            "month": [1],
            "ph": [np.nan],
        })
        test_data = val_data.copy()

        train_imp, val_imp, test_imp, _ = impute_missing_values(train_data, val_data, test_data, ["ph"])
        assert val_imp.loc[0, "ph"] == 7.4
        assert test_imp.loc[0, "ph"] == 7.4

    def test_standard_scaler_fitted_on_train_only(self):
        train_df = pd.DataFrame({"ph": [6.0, 7.0, 8.0]})
        val_df = pd.DataFrame({"ph": [7.0, 9.0]})
        test_df = pd.DataFrame({"ph": [5.0, 7.0]})

        train_sc, val_sc, test_sc, scaler = scale_features(train_df, val_df, test_df, ["ph"])

        assert np.isclose(train_sc["ph"].mean(), 0.0, atol=1e-6)
        assert np.isclose(train_sc["ph"].std(ddof=0), 1.0, atol=1e-6)
        # Val transformed with train mean (7.0) and std
        assert np.isclose(val_sc.loc[0, "ph"], 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 13. Summary, Output writing, and Real Dataset
# ---------------------------------------------------------------------------

class TestCleaningSummaryAndIO:

    def test_all_pipeline_stages_recorded(self, cleaned_fixture):
        _, summary = cleaned_fixture
        expected_steps = {
            "cast_data_types",
            "replace_sentinels",
            "fix_usa_temperature_unit",
            "clean_negative_concentrations",
            "drop_exact_duplicates",
            "flag_suspicious_values",
            "missing_value_strategy",
        }
        recorded_steps = {s["step"] for s in summary.steps}
        assert expected_steps.issubset(recorded_steps)

    def test_parquet_output_written(self, raw_fixture, tmp_path):
        raw_path = tmp_path / "raw.parquet"
        out_path = tmp_path / "cleaned.parquet"
        raw_fixture.to_parquet(str(raw_path), compression="snappy", index=False)

        pipeline = WaterQualityPreprocessor(raw_path=str(raw_path), output_path=str(out_path))
        pipeline.run()

        assert out_path.exists()
        reloaded = pd.read_parquet(str(out_path))
        assert len(reloaded) > 0


class TestRobustnessAndMetadataConsistency:

    def test_consistent_feature_order_across_partitions(self, cleaned_fixture):
        df_clean, _ = cleaned_fixture
        df_feat, _ = engineer_features(df_clean)
        df_large = pd.concat([df_feat] * 5, ignore_index=True)
        train_df, val_df, test_df = split_data(df_large)

        features = [f for f in MODEL_INPUT_FEATURES if f in train_df.columns]
        train_imp, val_imp, test_imp, _ = impute_missing_values(train_df, val_df, test_df, features)
        train_sc, val_sc, test_sc, _ = scale_features(train_imp, val_imp, test_imp, features)

        # Enforce exact identical feature column order across all partitions
        assert list(train_imp[features].columns) == features
        assert list(val_imp[features].columns) == features
        assert list(test_imp[features].columns) == features
        assert list(train_sc[features].columns) == features
        assert list(val_sc[features].columns) == features
        assert list(test_sc[features].columns) == features

    def test_empty_or_missing_file_raises_filenotfound(self):
        pipeline = WaterQualityPreprocessor(raw_path="non_existent_file.parquet")
        with pytest.raises(FileNotFoundError):
            pipeline.run()

    def test_malformed_unsupported_format_raises_valueerror(self, tmp_path):
        bad_file = tmp_path / "bad.txt"
        bad_file.write_text("random content")
        pipeline = WaterQualityPreprocessor(raw_path=str(bad_file))
        with pytest.raises(ValueError, match="Unsupported format"):
            pipeline.run()

    def test_prepare_training_datasets_creates_all_expected_artifacts(self, cleaned_fixture, tmp_path):
        df_clean, _ = cleaned_fixture
        df_large = pd.concat([df_clean] * 5, ignore_index=True)
        out_dir = tmp_path / "prep_output"

        res = prepare_training_datasets(
            cleaned_df=df_large,
            output_dir=str(out_dir),
            sample_size=10,
        )

        expected_files = [
            "train.parquet", "val.parquet", "test.parquet",
            "train_scaled.parquet", "val_scaled.parquet", "test_scaled.parquet",
            "train_sample.parquet", "val_sample.parquet", "test_sample.parquet",
            "preprocessing_metadata.json",
        ]
        for fname in expected_files:
            fpath = out_dir / fname
            assert fpath.exists(), f"Expected artifact '{fname}' was not generated"

        meta = res["metadata"]
        assert "metadata_version" in meta
        assert "input_features_ordered" in meta
        assert meta["input_features_ordered"] == MODEL_INPUT_FEATURES
        assert "scaler_parameters" in meta
        assert "leakage_safeguards" in meta

    def test_load_training_data_loader_isolation(self):
        sample_path = "data/GeneratedOutput/02.Preprocessing&FeatureSelection/train_sample.parquet"
        if os.path.exists(sample_path):
            X, y, feature_names = load_training_data(sample_path, target_col="ccme_values")
            assert X.shape == (50000, 15)
            assert y.shape == (50000,)
            assert feature_names == MODEL_INPUT_FEATURES
            assert np.isnan(X).sum() == 0
            assert np.isinf(X).sum() == 0
            # Test critical leakage detection: passing a target in feature_cols must raise ValueError
            with pytest.raises(ValueError, match="CRITICAL LEAKAGE DETECTED"):
                load_training_data(sample_path, feature_cols=MODEL_INPUT_FEATURES + ["ccme_values"])



REAL_DATASET_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "data", "Dataset", "Parquet", "Combined Data", "Combined_dataset.parquet",
)

@pytest.mark.skipif(
    not os.path.exists(REAL_DATASET_PATH),
    reason="Real dataset not available in test environment",
)
class TestRealDataset:

    def test_pipeline_on_real_dataset_first_10k(self, tmp_path):
        pipeline = WaterQualityPreprocessor(
            raw_path=REAL_DATASET_PATH,
            output_path=None,
            nrows=10_000,
        )
        df_out, summary = pipeline.run()

        assert_schema_preserved(df_out)
        assert_no_sentinels(df_out)
        assert_no_sdl_negatives(df_out)
        assert_no_exact_duplicates(df_out)
        assert_targets_complete(df_out)
        assert_flag_columns_present(df_out)
        assert_usa_temperatures_in_celsius_range(df_out)

        assert summary.raw_rows == 10_000
        assert summary.final_rows <= summary.raw_rows
