#!/usr/bin/env python3
"""
Phase 1 Presentation Deck Generator (PowerPoint)
Course: Semester 5 Applied Machine Learning (AML) | Project 18
Authors: 6-Member Team (Lead: Mayank Sharma)
File: reports/phase1_presentation.pptx
"""

import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "reports" / "phase1_presentation.pptx"

# Modern Deep Tech Dark Palette
COLOR_BG = RGBColor(11, 19, 37)        # Deep Navy (#0B1325)
COLOR_CARD = RGBColor(19, 31, 55)      # Dark Slate Card (#131F37)
COLOR_BORDER = RGBColor(30, 41, 59)    # Card Border (#1E293B)
TEXT_WHITE = RGBColor(248, 250, 252)   # Pure Bright White (#F8FAFC)
TEXT_MUTED = RGBColor(148, 163, 184)   # Slate Light Muted (#94A3B8)
ACCENT_CYAN = RGBColor(6, 182, 212)    # Electric Cyan (#06B6D4)
ACCENT_BLUE = RGBColor(59, 130, 246)   # Royal Blue (#3B82F6)
ACCENT_EMERALD = RGBColor(16, 185, 129)# Vibrant Emerald (#10B981)
ACCENT_GOLD = RGBColor(245, 158, 11)   # Warm Amber (#F59E0B)


def create_blank_slide(prs):
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = COLOR_BG
    bg.line.fill.background()
    return slide


def add_header(slide, title_text, category_text="PHASE 1 MACHINE LEARNING EVALUATION"):
    header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.733), Inches(0.9))
    tf = header_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p_cat = tf.paragraphs[0]
    p_cat.text = category_text.upper()
    p_cat.font.size = Pt(10)
    p_cat.font.bold = True
    p_cat.font.color.rgb = ACCENT_CYAN
    p_cat.space_after = Pt(2)

    p_title = tf.add_paragraph()
    p_title.text = title_text
    p_title.font.size = Pt(22)
    p_title.font.bold = True
    p_title.font.color.rgb = TEXT_WHITE


def add_card(slide, left, top, width, height, title, items, accent_color=ACCENT_BLUE):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = COLOR_CARD
    card.line.color.rgb = COLOR_BORDER
    card.line.width = Pt(1.5)

    tb = slide.shapes.add_textbox(left + Inches(0.25), top + Inches(0.2), width - Inches(0.5), height - Inches(0.4))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p0 = tf.paragraphs[0]
    p0.text = title
    p0.font.size = Pt(14)
    p0.font.bold = True
    p0.font.color.rgb = accent_color
    p0.space_after = Pt(8)

    for item in items:
        p = tf.add_paragraph()
        p.text = f"• {item}"
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT_WHITE
        p.space_after = Pt(4)


def add_metric_card(slide, left, top, width, height, value, label, subtext="", value_color=ACCENT_CYAN):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = COLOR_CARD
    card.line.color.rgb = COLOR_BORDER
    card.line.width = Pt(1.5)

    tb = slide.shapes.add_textbox(left + Inches(0.15), top + Inches(0.15), width - Inches(0.3), height - Inches(0.3))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p_val = tf.paragraphs[0]
    p_val.text = value
    p_val.font.size = Pt(22)
    p_val.font.bold = True
    p_val.font.color.rgb = value_color
    p_val.space_after = Pt(2)

    p_lbl = tf.add_paragraph()
    p_lbl.text = label
    p_lbl.font.size = Pt(11)
    p_lbl.font.bold = True
    p_lbl.font.color.rgb = TEXT_WHITE

    if subtext:
        p_sub = tf.add_paragraph()
        p_sub.text = subtext
        p_sub.font.size = Pt(9)
        p_sub.font.color.rgb = TEXT_MUTED


def build_phase1_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # -------------------------------------------------------------
    # SLIDE 1: Title & 6-Member Team Allocation
    # -------------------------------------------------------------
    s1 = create_blank_slide(prs)
    tb1 = s1.shapes.add_textbox(Inches(0.8), Inches(1.0), Inches(11.733), Inches(4.0))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p_top = tf1.paragraphs[0]
    p_top.text = "SEMESTER 5 APPLIED MACHINE LEARNING (AML) | PROJECT 18"
    p_top.font.size = Pt(12)
    p_top.font.bold = True
    p_top.font.color.rgb = ACCENT_CYAN
    p_top.space_after = Pt(10)

    p_head = tf1.add_paragraph()
    p_head.text = "Surface Water Quality Index Estimation\nvia Gradient Tree Boosting & Stoichiometry"
    p_head.font.size = Pt(30)
    p_head.font.bold = True
    p_head.font.color.rgb = TEXT_WHITE
    p_head.space_after = Pt(14)

    p_sub = tf1.add_paragraph()
    p_sub.text = "Phase 1 Machine Learning Evaluation: 2.82 Million Records Across 5 Nations"
    p_sub.font.size = Pt(16)
    p_sub.font.color.rgb = ACCENT_BLUE
    p_sub.space_after = Pt(20)

    # Team Attribution Box
    p_team = tf1.add_paragraph()
    p_team.text = "PROJECT TEAM (6 MEMBERS):"
    p_team.font.size = Pt(12)
    p_team.font.bold = True
    p_team.font.color.rgb = ACCENT_GOLD
    p_team.space_after = Pt(4)

    p_members = tf1.add_paragraph()
    p_members.text = (
        "• Mayank Sharma (Project Lead & AML Pipeline Architecture)\n"
        "• Team Member 2 (Literature Review, CCME WQI Math & Regulatory Frameworks)\n"
        "• Team Member 3 (Multi-National Dataset Auditing & Exploratory Data Analysis)\n"
        "• Team Member 4 (Aquatic Chemical Feature Engineering & Stoichiometric Physics)\n"
        "• Team Member 5 (Tree-Based Ensemble Optimization & Validation Benchmarking)\n"
        "• Team Member 6 (Explainable AI via TreeSHAP, Metric Verification & PyTest Suite)"
    )
    p_members.font.size = Pt(11)
    p_members.font.color.rgb = TEXT_WHITE

    # -------------------------------------------------------------
    # SLIDE 2: Problem Statement & Environmental Motivation
    # -------------------------------------------------------------
    s2 = create_blank_slide(prs)
    add_header(s2, "The Environmental Monitoring Crisis: Traditional Lab Delays vs. Real-Time AML")
    add_card(s2, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.4), "The Conventional Bottleneck", [
        "Freshwater Vulnerability: Global water systems face escalating nutrient runoff and industrial effluent contamination.",
        "Incubation Latency: Standard assays like 5-day Biochemical Oxygen Demand (BOD5) require 120 hours of incubation.",
        "High Analytical Cost: Grab-sampling requires specialized field personnel and recurring wet-lab operational budgets.",
        "Discrete Temporal Coverage: Infrequent monthly sampling misses rapid, transient toxic contamination plumes.",
        "Delayed Regulatory Intervention: By the time lab assays confirm eutrophication, ecological damage has occurred."
    ], ACCENT_GOLD)

    add_card(s2, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.4), "The Machine Learning Solution", [
        "In-Situ Instantaneous Telemetry: Leverage high-frequency physical and chemical probe telemetry (pH, DO, Temp, Conductivity).",
        "Deterministic Target Standardization: Predict the globally endorsed CCME Water Quality Index (CCME WQI, 0-100).",
        "Instantaneous Ecological Tiers: Map continuous scores directly into 5 regulatory tiers (Excellent to Poor).",
        "Sub-Second Response Time: Produce real-time index estimates in under 1 millisecond per sensor node.",
        "Scalable Automated Surveillance: Enables early warning flags for municipal intake plants and environmental regulators."
    ], ACCENT_CYAN)

    # -------------------------------------------------------------
    # SLIDE 3: Literature Review & Gap Analysis
    # -------------------------------------------------------------
    s3 = create_blank_slide(prs)
    add_header(s3, "Literature Review & Methodological Gap Analysis")
    add_card(s3, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.4), "Academic Foundations & Benchmarks", [
        "Breiman (2001) - Random Forests: Introduced bootstrap bagging with randomized candidate subspace splits; reduces variance and handles multi-modal tabular noise.",
        "Chen & Guestrin (2016) - XGBoost: Engineered second-order Taylor expansion gradients with leaf-weight regularization for high-speed tabular prediction.",
        "CCME (2001) - Water Quality Index 1.0: Established standard multi-dimensional aggregation across Scope (F1), Frequency (F2), and Amplitude (F3).",
        "Lundberg & Lee (2017) - TreeSHAP: Formulated exact polynomial-time Shapley attributions for tree ensembles grounded in cooperative game theory."
    ], ACCENT_BLUE)

    add_card(s3, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.4), "Critical Gaps in Existing Literature & Our Fixes", [
        "Gap 1: Unverified Telemetry Artifacts: Literature often ignores sensor telemetry errors. -> We enforce strict aquatic physics screening (purging 8,360 implausible values).",
        "Gap 2: Mathematical Target Leakage: Naive studies include intermediate exceedance components in features. -> We isolate a quarantined 38-feature predictor set.",
        "Gap 3: Extreme Nutrient Skewness: Raw orthophosphate spans 4 orders of magnitude. -> We implement log(1+x) transformations, boosting linear correlation from -0.66 to -0.85.",
        "Gap 4: Lack of Temporal Station Memory: Spatial models ignore local history. -> We engineer backward-only shift-1 lags (rho = -0.78 with WQI)."
    ], ACCENT_EMERALD)

    # -------------------------------------------------------------
    # SLIDE 4: Multi-National Dataset Profile
    # -------------------------------------------------------------
    s4 = create_blank_slide(prs)
    add_header(s4, "Multi-National Dataset Architecture: 2,827,977 Records Across 5 Nations")
    add_metric_card(s4, Inches(0.8), Inches(1.5), Inches(2.7), Inches(1.2), "2,827,977", "Total Ingested Records", "Historical archive 1940-2023", ACCENT_CYAN)
    add_metric_card(s4, Inches(3.8), Inches(1.5), Inches(2.7), Inches(1.2), "5 Nations", "Geographic Coverage", "England, USA, Ireland, China, Canada", ACCENT_BLUE)
    add_metric_card(s4, Inches(6.8), Inches(1.5), Inches(2.7), Inches(1.2), "100%", "Complete Case Ground Truth", "Zero null values in verified raw data", ACCENT_EMERALD)
    add_metric_card(s4, Inches(9.8), Inches(1.5), Inches(2.7), Inches(1.2), "83 Years", "Temporal Span", "Longitudinal aquatic monitoring", ACCENT_GOLD)

    img_path = REPO_ROOT / "data" / "GeneratedOutput" / "01.EDA" / "figures" / "01_country_distribution.png"
    if img_path.exists():
        s4.shapes.add_picture(str(img_path), Inches(0.8), Inches(2.9), width=Inches(6.0))

    add_card(s4, Inches(7.1), Inches(2.9), Inches(5.4), Inches(4.2), "Data Distribution Insights", [
        "England (Environment Agency): 2,367,978 observations (83.74%), providing rich river basin telemetry.",
        "United States (USGS Telemetry): 240,654 observations (8.51%), spanning major river catchments.",
        "Ireland (EPA Telemetry): 201,345 observations (7.13%), rich in high-purity lacustrine and river data.",
        "China & Canada: 18,000 observations (0.64%), benchmarking international cross-basin transferability.",
        "Waterbody Coverage: Rivers (88.4%), Lakes & Reservoirs (8.2%), Estuarine & Transitional Waters (3.4%)."
    ], ACCENT_CYAN)

    # -------------------------------------------------------------
    # SLIDE 5: Data Quality Audit & Physical Bounds Screening
    # -------------------------------------------------------------
    s5 = create_blank_slide(prs)
    add_header(s5, "Data Quality Audit: Purging 8,360 Physically Impossible Telemetry Errors")
    add_card(s5, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.4), "Thermodynamic & Chemical Screening Rules", [
        "pH Bounds [3.0, 12.0]: Natural waters cannot sustain pH < 3.0 (super-acidic mine drainage) or > 12.0 outside severe industrial caustic spills.",
        "Water Temperature [0.0°C, 45.0°C]: Constrained by the liquid state of surface freshwater under atmospheric pressure.",
        "Dissolved Oxygen [0.0, 25.0 mg/L]: Thermodynamic limit governed by Henry's law gas solubility and photosynthetic hyper-saturation.",
        "Non-Negative Nutrient Constraints: Conductivity, BOD5, Nitrate, Ammonia, and Orthophosphate cannot assume negative concentrations.",
        "Telemetry Error Codes: Purged sensor fault codes (-999, 9999) mistakenly recorded as numeric measurements."
    ], ACCENT_GOLD)

    add_card(s5, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.4), "Screening Audit Results", [
        "Raw Observations: 2,827,977 multi-country records.",
        "Physical Violations Purged: Exactly 8,360 records (0.296% of the dataset).",
        "Cleaned Production Dataset: 2,819,617 pristine observations with 100% complete case ground truth.",
        "Zero Synthetic Imputation: Avoided synthetic artifact introduction (e.g., KNN or mean imputation) by leveraging massive clean volume.",
        "Model Safety Impact: Eliminates unphysical split boundaries in decision trees, ensuring high-fidelity real-world generalization."
    ], ACCENT_EMERALD)

    # -------------------------------------------------------------
    # SLIDE 6: The CCME Water Quality Index Formulation
    # -------------------------------------------------------------
    s6 = create_blank_slide(prs)
    add_header(s6, "Target Definition: The CCME Water Quality Index (CCME WQI)")
    add_card(s6, Inches(0.8), Inches(1.5), Inches(6.0), Inches(5.4), "Mathematical Index Formulation", [
        "Composite Equation: WQI = 100 - [ sqrt(F1^2 + F2^2 + F3^2) / 1.732 ]",
        "F1 (Scope): Percentage of chemical variables that fail regulatory guidelines at least once: F1 = (Failed Variables / Total Variables) * 100",
        "F2 (Frequency): Percentage of individual tests failing regulatory thresholds: F2 = (Failed Tests / Total Tests) * 100",
        "F3 (Amplitude): Magnitude of exceedance via normalized sum of excursions (nse): nse = sum(excursion_i) / Total Tests",
        "Non-linear Scaling: F3 = nse / (0.01 * nse + 0.01); maps unbounded excursions smoothly into [0, 100]."
    ], ACCENT_CYAN)

    add_card(s6, Inches(7.1), Inches(1.5), Inches(5.4), Inches(5.4), "Standardized Regulatory Tiers", [
        "Tier 1: Excellent (95 - 100): Pristine ecological conditions; pristine water for all uses without treatment.",
        "Tier 2: Good (80 - 94): Minor degree of threat or impairment; conditions rarely depart from natural levels.",
        "Tier 3: Fair (65 - 79): Occasionally degraded; water quality occasionally departs from desirable levels.",
        "Tier 4: Marginal (45 - 64): Frequently threatened or impaired; conditions often depart from natural thresholds.",
        "Tier 5: Poor (0 - 44): Severely damaged; water quality is almost always impaired and fails regulatory criteria."
    ], ACCENT_BLUE)

    # -------------------------------------------------------------
    # SLIDE 7: Domain Feature Engineering: Aquatic Chemistry
    # -------------------------------------------------------------
    s7 = create_blank_slide(prs)
    add_header(s7, "Domain Feature Engineering: Geochemical & Stoichiometric Physics")
    add_card(s7, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.4), "Stoichiometric & Physical Formulations", [
        "Benson-Krause Dissolved Oxygen Saturation: Computes theoretical DO capacity DO_sat(T) as an empirical polynomial of temperature.",
        "DO Percent Saturation & Deficit: DO_%sat = (DO / DO_sat) * 100; DO_deficit = max(0, DO_sat - DO) reflecting biological oxygen demand.",
        "Redfield Stoichiometric Ratio: N:P = Nitrogen / (Orthophosphate + 1e-5), determining whether algal blooms are nitrogen- or phosphorus-limited.",
        "Emerson Un-Ionized Ammonia Equilibrium: Models lethal free NH3 as a function of aqueous temperature and pH.",
        "Streeter-Phelps Organic Load Ratio: BOD/DO ratio measuring microbial respiratory stress relative to available dissolved oxygen."
    ], ACCENT_CYAN)

    img_path7 = REPO_ROOT / "data" / "GeneratedOutput" / "01.EDA" / "figures" / "07_do_temperature_saturation.png"
    if img_path7.exists():
        s7.shapes.add_picture(str(img_path7), Inches(6.8), Inches(1.5), width=Inches(5.7))

    # -------------------------------------------------------------
    # SLIDE 8: Non-Linear Skewness Mitigation: The log(1+x) Transform
    # -------------------------------------------------------------
    s8 = create_blank_slide(prs)
    add_header(s8, "Non-Linear Skewness Mitigation: The log(1+x) Transformation")
    add_card(s8, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.4), "Distributional Skewness Challenge", [
        "Heavy Right Skew: Nutrients (Orthophosphate, Ammonia, BOD5) exhibit extreme skewness (gamma_1 > 40) across several orders of magnitude.",
        "Split Suboptimality: Decision trees partition skewed variables near the origin, wasting split capacity on extreme outliers.",
        "Mathematical Transformation: Applied y = ln(1 + x), which contracts heavy positive tails while preserving zero non-negativity.",
        "Dramatic Correlation Jump: Orthophosphate correlation with CCME WQI improved from r = -0.66 to r = -0.85.",
        "Improved Distance Metrics: Stabilizes gradient steps in gradient boosted ensembles and enhances tree partition efficiency."
    ], ACCENT_GOLD)

    img_path8 = REPO_ROOT / "data" / "GeneratedOutput" / "01.EDA" / "figures" / "08_correlation_heatmaps.png"
    if img_path8.exists():
        s8.shapes.add_picture(str(img_path8), Inches(6.8), Inches(1.5), width=Inches(5.7))

    # -------------------------------------------------------------
    # SLIDE 9: Chronological Station Memory (No Lookahead Leakage)
    # -------------------------------------------------------------
    s9 = create_blank_slide(prs)
    add_header(s9, "Spatial-Temporal Dynamics: Chronological Station Memory")
    add_card(s9, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.4), "Local Environmental Persistence", [
        "Hydrological Memory: Aquatic ecosystems exhibit temporal continuity; upstream contamination persists across successive sampling intervals.",
        "Shift-1 Station Lags: Computed lagged values (hist_do_lag1, hist_temp_lag1, hist_ortho_lag1) partitioned per monitoring station.",
        "Rolling-3 Statistics: Computed 3-sample rolling means capturing seasonal drift and localized baseline levels.",
        "Zero Lookahead Leakage: Stations were chronologically sorted; shift operations strictly look backward in time.",
        "High Predictive Correlation: Shift-1 orthophosphate lag (hist_ortho_lag1) achieves rank correlation rho = -0.78 with CCME WQI."
    ], ACCENT_BLUE)

    add_card(s9, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.4), "Temporal & Cyclical Harmonization", [
        "Harmonic Truncation: Encoded calendar Month and Day-of-Year as cyclical pairs: sin(2*pi*t/T) and cos(2*pi*t/T).",
        "Discontinuity Prevention: Ensures December 31 and January 1 remain continuous in distance space.",
        "Multi-Decadal Trend Tracking: Year feature tracks long-term legislative impacts (e.g., Clean Water Act, EU Water Framework Directive).",
        "Quarterly Grouping: Captures seasonal thermal turnover and agricultural fertilizer application pulses.",
        "Robust Geospatial Grouping: Categorical encoding for 5 nation codes and 3 major waterbody classes."
    ], ACCENT_EMERALD)

    # -------------------------------------------------------------
    # SLIDE 10: Strict Anti-Leakage Protocol & Splitting Strategy
    # -------------------------------------------------------------
    s10 = create_blank_slide(prs)
    add_header(s10, "Strict Anti-Leakage Protocol & Stratified Partitioning")
    add_card(s10, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.4), "Anti-Leakage Target Quarantine", [
        "The Circular Leakage Hazard: The CCME WQI is calculated from water parameters. Naive models including intermediate counters create circular leakage.",
        "Strict Quarantine Pipeline: All target columns (ccme_values, ccme_wqi, ccme_wqi_code, F1, F2, F3) are isolated via src.utils.quarantine_target.",
        "Feature Matrix Verification: Programmatic assertions verify 0 target variables exist inside the 38-feature matrix.",
        "Zero Post-Split Leakage: Scaling, normalization parameters, and metadata were fitted exclusively on the training partition.",
        "Automated Test Coverage: test_target_quarantine_function and test_metadata_feature_list_leakage_free enforce continuous CI verification."
    ], ACCENT_GOLD)

    add_card(s10, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.4), "Stratified 70/15/15 Data Partitioning", [
        "Dataset Volume: 2,819,617 cleaned records partitioned with exact class-stratification across ecological tiers.",
        "Training Set: 1,973,731 observations (70.0%) - sharded into transparent 3-part Parquet files under 60 MB for GitHub tracking.",
        "Validation Set: 422,943 observations (15.0%) - utilized for hyperparameter optimization and TreeSHAP calculations.",
        "Test Set: 422,943 observations (15.0%) - reserved for final held-out unbiased generalization benchmarking.",
        "Zero Distributional Shift: Stratification guarantees identical class proportions across all three partitions."
    ], ACCENT_CYAN)

    # -------------------------------------------------------------
    # SLIDE 11: Machine Learning Model Benchmarking
    # -------------------------------------------------------------
    s11 = create_blank_slide(prs)
    add_header(s11, "Phase 1 ML Model Benchmarks: Random Forest vs. XGBoost")
    
    img_path11 = REPO_ROOT / "data" / "GeneratedOutput" / "04.ModelEvaluation" / "model_comparison.png"
    if img_path11.exists():
        s11.shapes.add_picture(str(img_path11), Inches(0.8), Inches(1.5), width=Inches(6.0))

    add_card(s11, Inches(7.1), Inches(1.5), Inches(5.4), Inches(5.4), "Quantitative Empirical Performance", [
        "Mean Baseline: Val MAE = 14.130, RMSE = 17.656, R² = 0.0000. Provides empirical lower bound.",
        "Random Forest Regressor: Val MAE = 0.0521, RMSE = 0.2081, R² = 0.999861. Achieves highest precision through 100 de-correlated trees.",
        "XGBoost Regressor: Val MAE = 0.1375, RMSE = 0.3740, R² = 0.999551. Exceptional accuracy with 9x faster inference.",
        "Error Reduction: Both ensembles eliminate > 99.6% of baseline variance across 422,943 validation records.",
        "Deployment Recommendation: Random Forest for centralized offline auditing; XGBoost for low-latency edge sensor nodes."
    ], ACCENT_EMERALD)

    # -------------------------------------------------------------
    # SLIDE 12: Explainable AI (XAI) via TreeSHAP
    # -------------------------------------------------------------
    s12 = create_blank_slide(prs)
    add_header(s12, "Explainable AI (XAI): TreeSHAP Game-Theoretic Feature Attribution")
    
    img_path12 = REPO_ROOT / "data" / "GeneratedOutput" / "04.ModelEvaluation" / "shap_summary.png"
    if img_path12.exists():
        s12.shapes.add_picture(str(img_path12), Inches(0.8), Inches(1.5), width=Inches(6.0))

    add_card(s12, Inches(7.1), Inches(1.5), Inches(5.4), Inches(5.4), "Key Environmental Driver Discoveries", [
        "Orthophosphate Dominance: log1p_orthophosphate and raw orthophosphate account for > 88% of total split importance.",
        "Limiting Nutrient Mechanism: Phosphorus is freshwater's critical limiting nutrient; spikes trigger immediate CCME WQI collapse.",
        "Station Memory Ranks #2: hist_ortho_lag1 provides the second largest negative attribution, proving the value of local station history.",
        "Hypoxic Tipping Points: Dissolved oxygen exhibits steep negative SHAP attributions below 4.0 mg/L, reflecting fish mortality thresholds.",
        "Game-Theoretic Axioms: Shapley values guarantee Efficiency, Symmetry, Dummy player, and Additivity across all predictions."
    ], ACCENT_CYAN)

    # -------------------------------------------------------------
    # SLIDE 13: Modular Software Architecture & Automated Testing
    # -------------------------------------------------------------
    s13 = create_blank_slide(prs)
    add_header(s13, "Code Quality, Modularity & Automated Unit Testing Suite")
    add_card(s13, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.4), "Production Modular Package (src/)", [
        "src.features: filter_aquatic_physics, compute_log1p_features, compute_stoichiometry_ratios, compute_station_lags.",
        "src.models: DualHeadMLP model definitions and PyTorch MultiTaskLoss with dynamic loss balancing.",
        "src.metrics: Numerical evaluation functions for MAE, RMSE, R², Macro F1, Brier Score, and Expected Calibration Error (ECE).",
        "src.utils: load_parquet_dataset (transparent shard loading) and separate_features_and_targets (leakage prevention).",
        "Pipeline Runner: scripts/run_phase1_pipeline.py executes end-to-end data validation, inference, and test reporting."
    ], ACCENT_BLUE)

    add_card(s13, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.4), "12-Test Automated Test Suite (pytest)", [
        "Physics Bound Enforcement: test_filter_aquatic_physics confirms 100% detection of out-of-bounds telemetry.",
        "Transformation Accuracy: test_compute_log1p_features and test_compute_stoichiometry_ratios verify formulas.",
        "Strict Anti-Leakage Assertions: test_target_quarantine_function and test_metadata_feature_list_leakage_free verify zero target leakage.",
        "Model Serialization & Math: test_compute_regression_metrics, test_brier_score_and_ece, and test_dual_head_mlp_shapes.",
        "Execution Performance: 12 passed in 1.65 seconds (100% passing rate under pytest)."
    ], ACCENT_EMERALD)

    # -------------------------------------------------------------
    # SLIDE 14: Six-Member Team Contribution Matrix
    # -------------------------------------------------------------
    s14 = create_blank_slide(prs)
    add_header(s14, "Six-Member Project Allocation & Viva Voce Defense Matrix")
    
    # 2 columns x 3 rows of member cards
    members = [
        ("Mayank Sharma (Lead)", "Pipeline Architecture & Modeling", "Orchestrated end-to-end ML pipeline, Random Forest & XGBoost modeling, Parquet sharding, and GitHub versioning.", ACCENT_CYAN),
        ("Team Member 2", "Literature Review & Regulations", "Synthesized ML environmental literature, derived CCME WQI formulation (F1, F2, F3, nse), and international standards.", ACCENT_BLUE),
        ("Team Member 3", "Dataset Auditing & EDA", "Executed multi-national dataset audit across 5 countries, detected telemetry anomalies, and established physics boundaries.", ACCENT_EMERALD),
        ("Team Member 4", "Feature Engineering & Chemistry", "Engineered 38 features: Benson-Krause DO sat, Redfield N:P, Emerson toxic ammonia, log1p transforms, and station lags.", ACCENT_GOLD),
        ("Team Member 5", "Optimization & Benchmarking", "Managed hyperparameter optimization, loss functions, stratified 70/15/15 partitioning, and comparative benchmark metrics.", ACCENT_BLUE),
        ("Team Member 6", "Explainability (XAI) & Testing", "Conducted TreeSHAP attribution analysis, computed calibration metrics (Brier/ECE), and developed 12-test pytest suite.", ACCENT_CYAN),
    ]

    for i, (name, role, desc, color) in enumerate(members):
        col = i % 2
        row = i // 2
        l = Inches(0.8) if col == 0 else Inches(6.8)
        t = Inches(1.5 + row * 1.8)
        w = Inches(5.7)
        h = Inches(1.65)
        
        card = s14.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, l, t, w, h)
        card.fill.solid()
        card.fill.fore_color.rgb = COLOR_CARD
        card.line.color.rgb = COLOR_BORDER
        card.line.width = Pt(1.5)

        tb = s14.shapes.add_textbox(l + Inches(0.2), t + Inches(0.12), w - Inches(0.4), h - Inches(0.24))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

        p0 = tf.paragraphs[0]
        p0.text = f"{name} — {role}"
        p0.font.size = Pt(12)
        p0.font.bold = True
        p0.font.color.rgb = color
        p0.space_after = Pt(2)

        p1 = tf.add_paragraph()
        p1.text = desc
        p1.font.size = Pt(10)
        p1.font.color.rgb = TEXT_WHITE

    # -------------------------------------------------------------
    # SLIDE 15: Conclusion, Viva Voce Defense & Phase 2 Outlook
    # -------------------------------------------------------------
    s15 = create_blank_slide(prs)
    add_header(s15, "Phase 1 Key Conclusions, Viva Defense & Phase 2 Roadmap")
    add_card(s15, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.4), "Phase 1 Scientific Conclusions", [
        "Feasibility Proven: Tree ensembles predict continuous CCME WQI with R² > 0.999 across 2.82 million international records.",
        "Geochemical Feature Impact: Stoichiometry and log(1+x) normalization amplify signal, reducing tree depth requirements.",
        "Orthophosphate is Dominant: Game-theoretic TreeSHAP proves that limiting phosphorus drives over 88% of water quality index variations.",
        "Zero-Leakage Guarantee: Quarantined architecture prevents synthetic accuracy inflation.",
        "Enterprise Code Quality: Production src/ library validated by 12 unit tests passing in 1.65 seconds."
    ], ACCENT_EMERALD)

    add_card(s15, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.4), "Phase 2 Deep Learning Outlook & Discussion", [
        "Dual-Head Deep Neural Networks: Extending ML baselines to multi-task PyTorch MLP architectures (Huber loss + Cross-Entropy).",
        "Simultaneous Continuous & Discrete Outputs: Concurrently predicting continuous CCME values and 5-class ecological tiers.",
        "Probability Calibration & Uncertainty: Analyzing multi-class Brier score (0.0533) and Expected Calibration Error (1.27%).",
        "Edge Microservice Architecture: Containerizing inference engine in FastAPI for real-time municipal telemetry streams.",
        "Thank You! All 6 team members are prepared for individual viva examination."
    ], ACCENT_CYAN)

    prs.save(OUTPUT_PATH)
    print(f"✓ Phase 1 presentation saved successfully to: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_phase1_deck()
