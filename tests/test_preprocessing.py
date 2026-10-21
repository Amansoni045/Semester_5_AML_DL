"""
tests/test_preprocessing.py
============================
Unit tests for src/preprocessing.py

Test coverage:
  1. test_column_name_standardisation     — snake_case alias mapping, all 14 cols renamed
  2. test_original_column_mapping_preserved — COLUMN_NAME_MAP stored in attrs
  3. test_date_cast                        — string 'DD-MM-YYYY' -> datetime64
  4. test_ccme_wqi_ordered_categorical     — ordered Categorical dtype & order
  5. test_sentinel_replacement             — -999999, pH -1, pH 67115 -> NaN
  6. test_usa_temperature_correction       — F -> C for USA rows > 50
  7. test_non_usa_temperature_untouched    — England Effluent T > 50 NOT converted
  8. test_sdl_clipping                     — tiny negatives -> 0.0
  9. test_sdl_leaves_legitimate_negatives  — temperature stays negative
  10. test_exact_duplicates_dropped        — duplicate rows removed
  11. test_exact_duplicates_count          — correct number removed
  12. test_flag_columns_created            — all SUSPICIOUS_VALUE_RULES flagged
  13. test_flag_dtype_is_bool              — flags are boolean columns
  14. test_flag_not_nan                    — NaN rows are NOT flagged
  15. test_targets_remain_complete         — ccme_values, ccme_wqi have no NaN
  16. test_metadata_columns_intact         — country, area, waterbody_type, date preserved
  17. test_schema_assertion_helper         — assert_schema_preserved passes
  18. test_no_sentinels_assertion_helper   — assert_no_sentinels passes
  19. test_no_sdl_assertion_helper         — assert_no_sdl_negatives passes
  20. test_no_duplicates_assertion_helper  — assert_no_exact_duplicates passes
  21. test_targets_complete_assertion      — assert_targets_complete passes
  22. test_flag_assertion_helper           — assert_flag_columns_present passes
  23. test_usa_temp_assertion_helper       — assert_usa_temperatures_in_celsius_range
  24. test_cleaning_summary_step_count     — CleaningSummary records all pipeline stages
  25. test_cleaning_summary_row_counts     — raw_rows / final_rows populated correctly
  26. test_output_parquet_written          — file saved if output_path provided
  27. test_nrows_limit                     — nrows parameter restricts loaded rows
  28. test_full_pipeline_on_fixture        — end-to-end run on the small test fixture
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Project root on sys.path via conftest or direct adjustment
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import (
    COLUMN_NAME_MAP,
    CCME_WQI_CATEGORIES,
    MISSING_VALUE_STRATEGY,
    SDL_CLIP_COLUMNS,
    SUSPICIOUS_VALUE_RULES,
    CleaningSummary,
    WaterQualityPreprocessor,
    assert_flag_columns_present,
    assert_no_exact_duplicates,
    assert_no_sdl_negatives,
    assert_no_sentinels,
    assert_schema_preserved,
    assert_targets_complete,
    assert_usa_temperatures_in_celsius_range,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_raw_fixture(n_extra_duplicates: int = 2) -> pd.DataFrame:
    """
    Build a minimal DataFrame that mirrors the raw combined dataset schema
    (original column names, original dtypes) with intentional quality issues.

    Issues injected:
      - Sentinel temperature: -999999.0
      - pH = -999999.0 (sentinel)
      - pH = -1.0
      - pH = 67115.0 (scale corruption)
      - pH = 15.0 (implausible)
      - USA temperature > 50 (Fahrenheit)
      - SDL negatives in Ammonia and BOD
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

    # Inject -999999 sentinel into row 0 temperature
    df.loc[0, "Temperature (cel)"] = -999999.0
    # Inject -999999 into DO row 1
    df.loc[1, "Dissolved Oxygen (mg/l)"] = -999999.0

    # Add explicit duplicate rows (exact copy of row 4)
    dup_rows = pd.concat([df.iloc[[4]]] * n_extra_duplicates, ignore_index=True)
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df


@pytest.fixture
def raw_fixture() -> pd.DataFrame:
    """Small raw DataFrame with known quality issues."""
    return _make_raw_fixture(n_extra_duplicates=2)


@pytest.fixture
def cleaned_fixture(raw_fixture, tmp_path) -> tuple:
    """Run pipeline on raw_fixture and return (cleaned_df, summary)."""
    # Save raw fixture as a temporary Parquet to use with the pipeline
    raw_path = tmp_path / "raw_fixture.parquet"
    raw_fixture.to_parquet(str(raw_path), compression="snappy", index=False)

    pipeline = WaterQualityPreprocessor(
        raw_path=str(raw_path),
        output_path=None,  # no save
    )
    return pipeline.run()


# ---------------------------------------------------------------------------
# 1. Column name standardisation
# ---------------------------------------------------------------------------

class TestColumnNameStandardisation:

    def test_all_original_columns_renamed(self, cleaned_fixture):
        df, _ = cleaned_fixture
        # None of the original long names should remain
        for original_col in COLUMN_NAME_MAP.keys():
            assert original_col not in df.columns, (
                f"Original column '{original_col}' should have been renamed"
            )

    def test_all_alias_columns_present(self, cleaned_fixture):
        df, _ = cleaned_fixture
        for alias in COLUMN_NAME_MAP.values():
            assert alias in df.columns, f"Alias column '{alias}' missing after rename"

    def test_original_column_mapping_in_attrs(self, cleaned_fixture):
        df, _ = cleaned_fixture
        assert "column_mapping" in df.attrs, "df.attrs should contain 'column_mapping'"
        mapping = df.attrs["column_mapping"]
        assert mapping == COLUMN_NAME_MAP


# ---------------------------------------------------------------------------
# 2. Data type casting
# ---------------------------------------------------------------------------

class TestDataTypeCasting:

    def test_date_cast_to_datetime(self, cleaned_fixture):
        df, _ = cleaned_fixture
        assert pd.api.types.is_datetime64_any_dtype(df["date"]), (
            f"'date' should be datetime64, got {df['date'].dtype}"
        )

    def test_date_parsed_correctly(self, cleaned_fixture):
        df, _ = cleaned_fixture
        # "15-03-2010" -> 2010-03-15
        assert df["date"].dropna().iloc[0].year == 2010

    def test_ccme_wqi_is_ordered_categorical(self, cleaned_fixture):
        df, _ = cleaned_fixture
        assert hasattr(df["ccme_wqi"], "cat"), "'ccme_wqi' should be Categorical"
        assert df["ccme_wqi"].cat.ordered, "'ccme_wqi' should be an ordered Categorical"

    def test_ccme_wqi_category_order(self, cleaned_fixture):
        df, _ = cleaned_fixture
        cats = list(df["ccme_wqi"].cat.categories)
        # Verify order: Poor < Marginal < Fair < Good < Excellent
        expected = [c for c in CCME_WQI_CATEGORIES if c in cats]
        assert cats == expected, f"Category order wrong: {cats}"

    def test_numeric_columns_are_float(self, cleaned_fixture):
        df, _ = cleaned_fixture
        numeric_cols = [
            "ammonia_mg_l", "bod_mg_l", "do_mg_l", "orthophosphate_mg_l",
            "ph", "temperature_c", "nitrogen_mg_l", "nitrate_mg_l", "ccme_values",
        ]
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(df[col]), (
                f"'{col}' should be numeric, got {df[col].dtype}"
            )


# ---------------------------------------------------------------------------
# 3. Sentinel replacement
# ---------------------------------------------------------------------------

class TestSentinelReplacement:

    def test_sentinel_temperature_replaced_with_nan(self, cleaned_fixture):
        df, _ = cleaned_fixture
        # Row 0 had Temperature = -999999.0 -> should be NaN
        assert pd.isna(df["temperature_c"].iloc[0]), (
            "Sentinel temperature -999999 should have been replaced with NaN"
        )

    def test_sentinel_do_replaced_with_nan(self, cleaned_fixture):
        df, _ = cleaned_fixture
        # Row 1 had DO = -999999.0 -> should be NaN
        # (after dedup rows shift, check that the value exists as NaN somewhere)
        assert int((df["do_mg_l"] <= -999).sum()) == 0, (
            "All sentinel DO values should be replaced with NaN"
        )

    def test_ph_minus1_replaced_with_nan(self, cleaned_fixture):
        df, _ = cleaned_fixture
        assert int((df["ph"] == -1.0).sum()) == 0, (
            "pH = -1.0 should have been replaced with NaN"
        )

    def test_ph_scale_corruption_replaced(self, cleaned_fixture):
        df, _ = cleaned_fixture
        assert int((df["ph"] == 67115.0).sum()) == 0, (
            "pH = 67115.0 (scale corruption) should have been replaced with NaN"
        )

    def test_ph_above_14_replaced_with_nan(self, cleaned_fixture):
        df, _ = cleaned_fixture
        # pH = 15.0 and pH = 67115.0 both above 14
        n_ph_high = int((df["ph"] > 14).sum())
        assert n_ph_high == 0, f"pH > 14.0 still present: {n_ph_high} rows"

    def test_no_extreme_negatives_remain(self, cleaned_fixture):
        df, _ = cleaned_fixture
        assert_no_sentinels(df)


# ---------------------------------------------------------------------------
# 4. USA temperature correction
# ---------------------------------------------------------------------------

class TestUSATemperatureCorrection:

    def test_usa_temperatures_converted(self, cleaned_fixture):
        df, _ = cleaned_fixture
        # USA rows had Temperature = 68.0 (F) -> should now be (68-32)*5/9 = 20.0 C
        usa_rows = df[df["country"] == "USA"]
        assert all(usa_rows["temperature_c"] <= 50), (
            "All USA temperatures should be <= 50C after Fahrenheit correction"
        )

    def test_conversion_value_correct(self, cleaned_fixture):
        df, _ = cleaned_fixture
        # 68F -> 20.0C exactly
        usa_rows = df[df["country"] == "USA"]
        expected_c = (68.0 - 32) * 5 / 9
        actual_temps = usa_rows["temperature_c"].dropna().values
        assert all(abs(t - expected_c) < 0.01 for t in actual_temps), (
            f"Expected {expected_c:.2f}C, got {actual_temps}"
        )

    def test_non_usa_high_temp_not_converted(self, raw_fixture, tmp_path):
        """England Effluent rows with T > 50 should NOT be converted."""
        # Inject an England Effluent row with T = 60
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
        assert any(eng_eff["temperature_c"] == 60.0), (
            "England Effluent temperature of 60C should NOT be converted"
        )

    def test_assertion_helper_passes(self, cleaned_fixture):
        df, _ = cleaned_fixture
        assert_usa_temperatures_in_celsius_range(df)  # should not raise


# ---------------------------------------------------------------------------
# 5. Sub-detection-limit negative clipping
# ---------------------------------------------------------------------------

class TestSDLClipping:

    def test_ammonia_sdl_clipped(self, cleaned_fixture):
        df, _ = cleaned_fixture
        assert int((df["ammonia_mg_l"] < 0).sum()) == 0, (
            "Ammonia should have no negative values after SDL clipping"
        )

    def test_bod_sdl_clipped(self, cleaned_fixture):
        df, _ = cleaned_fixture
        assert int((df["bod_mg_l"] < 0).sum()) == 0, (
            "BOD should have no negative values after SDL clipping"
        )

    def test_do_sdl_clipped(self, cleaned_fixture):
        df, _ = cleaned_fixture
        assert int((df["do_mg_l"] < 0).sum()) == 0, (
            "DO should have no negative values after SDL clipping"
        )

    def test_orthophosphate_sdl_clipped(self, cleaned_fixture):
        df, _ = cleaned_fixture
        assert int((df["orthophosphate_mg_l"] < 0).sum()) == 0, (
            "Orthophosphate should have no negative values after SDL clipping"
        )

    def test_assertion_helper_passes(self, cleaned_fixture):
        df, _ = cleaned_fixture
        assert_no_sdl_negatives(df)

    def test_legitimate_negatives_in_temperature_untouched(self, tmp_path):
        """Temperature can legitimately go sub-zero; SDL clipping should not touch it."""
        raw = _make_raw_fixture()
        raw.loc[0, "Temperature (cel)"] = -2.5  # legitimate cold temp
        raw_path = tmp_path / "cold_temp.parquet"
        raw.to_parquet(str(raw_path), compression="snappy", index=False)

        pipeline = WaterQualityPreprocessor(raw_path=str(raw_path))
        df_out, _ = pipeline.run()
        assert any(df_out["temperature_c"] == -2.5), (
            "Legitimate sub-zero temperature should NOT be clipped to 0.0"
        )


# ---------------------------------------------------------------------------
# 6. Exact duplicate removal
# ---------------------------------------------------------------------------

class TestDuplicateRemoval:

    def test_no_exact_duplicates_remain(self, cleaned_fixture):
        df, _ = cleaned_fixture
        assert_no_exact_duplicates(df)

    def test_correct_number_of_duplicates_removed(self, raw_fixture, tmp_path):
        """2 extra duplicates of row 4 injected -> 2 should be removed."""
        raw_path = tmp_path / "dup_fixture.parquet"
        raw_fixture.to_parquet(str(raw_path), compression="snappy", index=False)

        n_raw = len(raw_fixture)
        n_exact_dups = int(raw_fixture.duplicated().sum())

        pipeline = WaterQualityPreprocessor(raw_path=str(raw_path))
        df_out, summary = pipeline.run()

        dedup_step = next(
            (s for s in summary.steps if s["step"] == "drop_exact_duplicates"), None
        )
        assert dedup_step is not None, "drop_exact_duplicates step not found in summary"
        assert dedup_step["rows_removed"] == n_exact_dups, (
            f"Expected {n_exact_dups} duplicates removed, got {dedup_step['rows_removed']}"
        )

    def test_non_duplicate_rows_retained(self, cleaned_fixture, raw_fixture):
        df, _ = cleaned_fixture
        # Original unique rows: 7 (indices 0-6). Duplicates: 2 extra copies of row 4.
        # After dedup: 7 unique rows remain (minus sentinels don't drop rows).
        n_unique_raw = len(raw_fixture.drop_duplicates())
        # The pipeline processes all of them; no additional rows should be dropped
        assert len(df) == n_unique_raw, (
            f"Expected {n_unique_raw} rows (unique), got {len(df)}"
        )


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
            if flag_col in df.columns:
                assert df[flag_col].dtype == bool, (
                    f"Flag column '{flag_col}' should be bool, got {df[flag_col].dtype}"
                )

    def test_nan_values_not_flagged(self, cleaned_fixture):
        """NaN values should NOT be flagged - only implausible non-null values."""
        df, _ = cleaned_fixture
        for col in SUSPICIOUS_VALUE_RULES:
            flag_col = f"{col}_flag_suspicious"
            if flag_col not in df.columns or col not in df.columns:
                continue
            # Where the original column is NaN, the flag must be False
            nan_mask = df[col].isna()
            if nan_mask.any():
                assert not df.loc[nan_mask, flag_col].any(), (
                    f"Flag '{flag_col}' is True for NaN rows — NaN should never be flagged"
                )

    def test_flagging_does_not_remove_rows(self, raw_fixture, tmp_path):
        """Flagging stage should add columns but NOT remove any rows."""
        raw_path = tmp_path / "flag_test.parquet"
        raw_fixture.to_parquet(str(raw_path), compression="snappy", index=False)

        pipeline = WaterQualityPreprocessor(raw_path=str(raw_path))
        df_out, summary = pipeline.run()

        flag_step = next(
            (s for s in summary.steps if s["step"] == "flag_suspicious_values"), None
        )
        assert flag_step is not None
        assert flag_step["rows_removed"] == 0, (
            "Flagging stage should not remove any rows"
        )


# ---------------------------------------------------------------------------
# 8. Target completeness
# ---------------------------------------------------------------------------

class TestTargetCompleteness:

    def test_ccme_values_no_missing(self, cleaned_fixture):
        df, _ = cleaned_fixture
        assert df["ccme_values"].isna().sum() == 0

    def test_ccme_wqi_no_missing(self, cleaned_fixture):
        df, _ = cleaned_fixture
        assert df["ccme_wqi"].isna().sum() == 0

    def test_assertion_helper_passes(self, cleaned_fixture):
        df, _ = cleaned_fixture
        assert_targets_complete(df)


# ---------------------------------------------------------------------------
# 9. Metadata column preservation
# ---------------------------------------------------------------------------

class TestMetadataPreservation:

    def test_country_column_preserved(self, cleaned_fixture):
        df, _ = cleaned_fixture
        assert "country" in df.columns
        assert df["country"].notna().all()

    def test_area_column_preserved(self, cleaned_fixture):
        df, _ = cleaned_fixture
        assert "area" in df.columns
        assert df["area"].notna().all()

    def test_waterbody_type_preserved(self, cleaned_fixture):
        df, _ = cleaned_fixture
        assert "waterbody_type" in df.columns

    def test_date_column_preserved(self, cleaned_fixture):
        df, _ = cleaned_fixture
        assert "date" in df.columns

    def test_schema_assertion_helper_passes(self, cleaned_fixture):
        df, _ = cleaned_fixture
        assert_schema_preserved(df)


# ---------------------------------------------------------------------------
# 10. CleaningSummary
# ---------------------------------------------------------------------------

class TestCleaningSummary:

    def test_all_pipeline_stages_recorded(self, cleaned_fixture):
        _, summary = cleaned_fixture
        expected_steps = {
            "cast_data_types",
            "replace_sentinels",
            "fix_usa_temperature_unit",
            "clip_sdl_negatives",
            "drop_exact_duplicates",
            "flag_suspicious_values",
            "missing_value_strategy",
        }
        recorded_steps = {s["step"] for s in summary.steps}
        assert expected_steps.issubset(recorded_steps), (
            f"Missing summary steps: {expected_steps - recorded_steps}"
        )

    def test_raw_rows_populated(self, raw_fixture, tmp_path):
        raw_path = tmp_path / "summary_test.parquet"
        raw_fixture.to_parquet(str(raw_path), compression="snappy", index=False)
        pipeline = WaterQualityPreprocessor(raw_path=str(raw_path))
        _, summary = pipeline.run()
        assert summary.raw_rows == len(raw_fixture)

    def test_final_rows_less_than_raw(self, cleaned_fixture, raw_fixture):
        _, summary = cleaned_fixture
        assert summary.final_rows < summary.raw_rows, (
            "Deduplication should reduce row count"
        )

    def test_flags_added_list_populated(self, cleaned_fixture):
        _, summary = cleaned_fixture
        assert len(summary.flags_added) > 0

    def test_summary_to_dataframe_shape(self, cleaned_fixture):
        _, summary = cleaned_fixture
        df_steps = summary.to_dataframe()
        assert isinstance(df_steps, pd.DataFrame)
        assert len(df_steps) == len(summary.steps)
        assert "step" in df_steps.columns
        assert "rows_before" in df_steps.columns
        assert "rows_after" in df_steps.columns
        assert "rows_removed" in df_steps.columns


# ---------------------------------------------------------------------------
# 11. Output file writing
# ---------------------------------------------------------------------------

class TestOutputFile:

    def test_parquet_written_when_output_path_provided(self, raw_fixture, tmp_path):
        raw_path = tmp_path / "raw.parquet"
        out_path = tmp_path / "cleaned.parquet"
        raw_fixture.to_parquet(str(raw_path), compression="snappy", index=False)

        pipeline = WaterQualityPreprocessor(
            raw_path=str(raw_path),
            output_path=str(out_path),
        )
        pipeline.run()

        assert out_path.exists(), "Output Parquet file was not created"
        df_reload = pd.read_parquet(str(out_path))
        assert len(df_reload) > 0

    def test_no_file_written_when_output_path_is_none(self, raw_fixture, tmp_path):
        raw_path = tmp_path / "raw.parquet"
        raw_fixture.to_parquet(str(raw_path), compression="snappy", index=False)

        pipeline = WaterQualityPreprocessor(raw_path=str(raw_path), output_path=None)
        pipeline.run()

        # No .parquet files should appear in tmp_path except raw.parquet
        parquet_files = list(tmp_path.glob("*.parquet"))
        assert len(parquet_files) == 1, (
            f"Expected only raw.parquet in tmp_path, found: {parquet_files}"
        )


# ---------------------------------------------------------------------------
# 12. nrows parameter
# ---------------------------------------------------------------------------

class TestNrowsParameter:

    def test_nrows_limits_loaded_rows(self, raw_fixture, tmp_path):
        raw_path = tmp_path / "raw.parquet"
        raw_fixture.to_parquet(str(raw_path), compression="snappy", index=False)

        pipeline = WaterQualityPreprocessor(raw_path=str(raw_path), nrows=3)
        df_out, summary = pipeline.run()

        assert summary.raw_rows == 3, (
            f"Expected raw_rows=3, got {summary.raw_rows}"
        )


# ---------------------------------------------------------------------------
# 13. Full pipeline end-to-end on fixture
# ---------------------------------------------------------------------------

class TestFullPipelineOnFixture:

    def test_end_to_end_no_exceptions(self, raw_fixture, tmp_path):
        """Pipeline should run without raising any exception on the test fixture."""
        raw_path = tmp_path / "raw.parquet"
        raw_fixture.to_parquet(str(raw_path), compression="snappy", index=False)
        pipeline = WaterQualityPreprocessor(raw_path=str(raw_path))
        df_out, summary = pipeline.run()
        assert df_out is not None
        assert summary is not None

    def test_missing_value_strategy_in_attrs(self, cleaned_fixture):
        df, _ = cleaned_fixture
        assert "missing_value_strategy" in df.attrs
        strategy = df.attrs["missing_value_strategy"]
        assert "ammonia_mg_l" in strategy
        assert "ph" in strategy
        assert "ccme_values" in strategy

    def test_source_path_in_attrs(self, raw_fixture, tmp_path):
        raw_path = tmp_path / "raw.parquet"
        raw_fixture.to_parquet(str(raw_path), compression="snappy", index=False)
        pipeline = WaterQualityPreprocessor(raw_path=str(raw_path))
        df_out, _ = pipeline.run()
        assert "source_path" in df_out.attrs
        assert df_out.attrs["source_path"] == str(raw_path)

    def test_cleaned_df_has_correct_columns(self, cleaned_fixture):
        df, _ = cleaned_fixture
        # Should have original 14 aliased columns + flag columns
        original_aliases = set(COLUMN_NAME_MAP.values())
        assert original_aliases.issubset(set(df.columns)), (
            f"Missing columns: {original_aliases - set(df.columns)}"
        )


# ---------------------------------------------------------------------------
# 14. Integration: run on real dataset if available (optional / skipped in CI)
# ---------------------------------------------------------------------------

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
        """Run the full pipeline on the first 10,000 rows of the real dataset."""
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

    def test_real_dataset_full_run_schema(self, tmp_path):
        """Run on the full real dataset and validate schema only (slow test)."""
        out_path = tmp_path / "cleaned_real.parquet"
        pipeline = WaterQualityPreprocessor(
            raw_path=REAL_DATASET_PATH,
            output_path=str(out_path),
        )
        df_out, summary = pipeline.run()

        assert_schema_preserved(df_out)
        assert_no_sentinels(df_out)
        assert_no_sdl_negatives(df_out)
        assert_targets_complete(df_out)
        assert out_path.exists()
