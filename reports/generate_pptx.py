#!/usr/bin/env python3
"""
Generate a professional, publication-quality 18-slide PowerPoint presentation (.pptx)
for the Water Quality Index AML & DL project.
"""

import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# --- Color Palette (Dark Theme / Slate & Modern Navy) ---
BG_COLOR = RGBColor(11, 19, 37)         # Deep Navy / Slate (#0B1325)
CARD_BG = RGBColor(26, 38, 57)          # Card Slate Navy (#1A2639)
CARD_BORDER = RGBColor(45, 62, 80)      # Border (#2D3E50)
TEXT_WHITE = RGBColor(248, 250, 252)    # Pure White (#F8FAFC)
TEXT_MUTED = RGBColor(148, 163, 184)    # Slate Muted (#94A3B8)
ACCENT_BLUE = RGBColor(56, 189, 248)    # Electric Sky Blue (#38BDF8)
ACCENT_CYAN = RGBColor(45, 212, 191)    # Bright Cyan/Teal (#2DD4BF)
ACCENT_EMERALD = RGBColor(16, 185, 129) # Emerald Green (#10B981)
ACCENT_AMBER = RGBColor(245, 158, 11)   # Amber Orange (#F59E0B)
ACCENT_ROSE = RGBColor(244, 63, 94)     # Rose Red (#F43F5E)

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(REPO_DIR, "reports", "water_quality_presentation.pptx")

def get_fig(rel_path):
    p = os.path.join(REPO_DIR, rel_path)
    if os.path.exists(p):
        return p
    return None

def create_presentation():
    prs = Presentation()
    # 16:9 widescreen layout (13.333 x 7.5 inches)
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6] # Blank slide layout

    def set_background(slide):
        bg = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height
        )
        bg.fill.solid()
        bg.fill.fore_color.rgb = BG_COLOR
        bg.line.color.rgb = BG_COLOR
        return bg

    def add_header(slide, title, category="SEMESTER 5 AML & DL | PROJECT 18", slide_num=None):
        set_background(slide)
        
        # Category / Tracker Pill
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(9.0), Inches(0.35))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        tf_cat.margin_left = tf_cat.margin_right = tf_cat.margin_top = tf_cat.margin_bottom = 0
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category.upper()
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = ACCENT_BLUE

        # Main Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.72), Inches(10.5), Inches(0.6))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        tf_title.margin_left = tf_title.margin_right = tf_title.margin_top = tf_title.margin_bottom = 0
        p_title = tf_title.paragraphs[0]
        p_title.text = title
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = TEXT_WHITE

        # Divider line
        line = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.35), Inches(11.733), Inches(0.02)
        )
        line.fill.solid()
        line.fill.fore_color.rgb = CARD_BORDER
        line.line.color.rgb = CARD_BORDER

        # Slide Number
        if slide_num is not None:
            num_box = slide.shapes.add_textbox(Inches(11.5), Inches(0.55), Inches(1.0), Inches(0.4))
            tf_num = num_box.text_frame
            tf_num.margin_left = tf_num.margin_right = tf_num.margin_top = tf_num.margin_bottom = 0
            p_num = tf_num.paragraphs[0]
            p_num.alignment = PP_ALIGN.RIGHT
            p_num.text = f"{slide_num:02d} / 18"
            p_num.font.size = Pt(11)
            p_num.font.bold = True
            p_num.font.color.rgb = TEXT_MUTED

    def add_card(slide, left, top, width, height, title=None, bullets=None, accent_color=ACCENT_BLUE, fill_color=CARD_BG):
        # Card shape
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = fill_color
        card.line.color.rgb = CARD_BORDER
        card.line.width = Pt(1.2)

        # Content box inside card
        tb = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.2), width - Inches(0.4), height - Inches(0.4))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0

        p_first = tf.paragraphs[0]
        if title:
            p_first.text = title
            p_first.font.size = Pt(14)
            p_first.font.bold = True
            p_first.font.color.rgb = accent_color
            p_first.space_after = Pt(8)
        
        if bullets:
            for i, b in enumerate(bullets):
                p = tf.add_paragraph() if (title or i > 0) else p_first
                p.text = f"• {b}"
                p.font.size = Pt(11)
                p.font.color.rgb = TEXT_WHITE
                p.space_after = Pt(4)
        return card

    def add_metric_card(slide, left, top, width, height, val, label, subtext=None, accent_color=ACCENT_BLUE):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = CARD_BORDER
        card.line.width = Pt(1.2)

        tb = slide.shapes.add_textbox(left + Inches(0.15), top + Inches(0.15), width - Inches(0.3), height - Inches(0.3))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0

        # Metric value
        p_val = tf.paragraphs[0]
        p_val.text = val
        p_val.font.size = Pt(24)
        p_val.font.bold = True
        p_val.font.color.rgb = accent_color
        p_val.space_after = Pt(2)

        # Label
        p_lbl = tf.add_paragraph()
        p_lbl.text = label
        p_lbl.font.size = Pt(12)
        p_lbl.font.bold = True
        p_lbl.font.color.rgb = TEXT_WHITE
        p_lbl.space_after = Pt(2)

        # Subtext
        if subtext:
            p_sub = tf.add_paragraph()
            p_sub.text = subtext
            p_sub.font.size = Pt(9.5)
            p_sub.font.color.rgb = TEXT_MUTED

    def add_table_data(slide, left, top, width, height, headers, rows, col_widths=None):
        table_shape = slide.shapes.add_table(len(rows) + 1, len(headers), left, top, width, height)
        table = table_shape.table
        
        if col_widths:
            for i, w in enumerate(col_widths):
                table.columns[i].width = w

        for col_idx, h in enumerate(headers):
            cell = table.cell(0, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(30, 48, 77)
            cell.text = h
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(11)
            p.font.bold = True
            p.font.color.rgb = ACCENT_BLUE
            p.alignment = PP_ALIGN.CENTER

        for row_idx, r in enumerate(rows):
            bg = CARD_BG if row_idx % 2 == 0 else RGBColor(20, 31, 48)
            for col_idx, val in enumerate(r):
                cell = table.cell(row_idx + 1, col_idx)
                cell.fill.solid()
                cell.fill.fore_color.rgb = bg
                cell.text = str(val)
                p = cell.text_frame.paragraphs[0]
                p.font.size = Pt(10)
                p.font.color.rgb = TEXT_WHITE
                p.alignment = PP_ALIGN.CENTER if col_idx > 0 else PP_ALIGN.LEFT

    # ==========================================
    # SLIDE 1: Title Slide
    # ==========================================
    s1 = prs.slides.add_slide(blank_layout)
    set_background(s1)

    # Accent decorative banner
    banner = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.8), Inches(0.12), Inches(3.2))
    banner.fill.solid()
    banner.fill.fore_color.rgb = ACCENT_BLUE
    banner.line.color.rgb = ACCENT_BLUE

    # Title Box
    tbox = s1.shapes.add_textbox(Inches(1.2), Inches(1.8), Inches(11.0), Inches(3.2))
    tf1 = tbox.text_frame
    tf1.word_wrap = True
    tf1.margin_left = tf1.margin_right = tf1.margin_top = tf1.margin_bottom = 0

    p_badge = tf1.paragraphs[0]
    p_badge.text = "SEMESTER 5 AML & DL CAPSTONE PROJECT • 2.82M OBSERVATIONS"
    p_badge.font.size = Pt(12)
    p_badge.font.bold = True
    p_badge.font.color.rgb = ACCENT_CYAN
    p_badge.space_after = Pt(10)

    p_main = tf1.add_paragraph()
    p_main.text = "Water Quality Index Prediction & Classification"
    p_main.font.size = Pt(36)
    p_main.font.bold = True
    p_main.font.color.rgb = TEXT_WHITE
    p_main.space_after = Pt(6)

    p_sub = tf1.add_paragraph()
    p_sub.text = "Ensemble Gradient Boosting vs. Dual-Head Deep Neural Networks Across 5 Nations"
    p_sub.font.size = Pt(18)
    p_sub.font.color.rgb = ACCENT_BLUE
    p_sub.space_after = Pt(18)

    p_auth = tf1.add_paragraph()
    p_auth.text = "Author: Mayank Sharma  |  GitHub: @MAYANKSHARMA01010/Semester_5_AML_DL  |  October 2026"
    p_auth.font.size = Pt(12)
    p_auth.font.color.rgb = TEXT_MUTED

    # Bottom metric ribbon
    add_metric_card(s1, Inches(0.8), Inches(5.4), Inches(2.7), Inches(1.4), "2,819,617", "Clean Records", "Complete Case Ground Truth", ACCENT_CYAN)
    add_metric_card(s1, Inches(3.8), Inches(5.4), Inches(2.7), Inches(1.4), "0.9999", "Random Forest R²", "Val MAE: 0.0521 Index Pts", ACCENT_EMERALD)
    add_metric_card(s1, Inches(6.8), Inches(5.4), Inches(2.7), Inches(1.4), "96.09%", "Dual-Head DL Acc", "97.49% Recall on 'Poor' Tier", ACCENT_BLUE)
    add_metric_card(s1, Inches(9.8), Inches(5.4), Inches(2.7), Inches(1.4), "38 Features", "Engineered Inputs", "Log1p + Lags + Guidance Stress", ACCENT_AMBER)

    # ==========================================
    # SLIDE 2: Executive Summary & Project Impact
    # ==========================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Executive Summary: Environmental Motivation & Key Findings", slide_num=2)

    add_card(s2, Inches(0.8), Inches(1.6), Inches(5.6), Inches(2.5), "The Problem: Latency in Water Testing", [
        "Traditional water testing relies on physical grab sampling and multi-day laboratory assays (e.g., 5-day BOD assays).",
        "Delayed contamination reporting prevents rapid municipal response to toxic runoff and industrial chemical spills.",
        "Regulatory water indices like CCME WQI are mathematically rigorous but cumbersome to compute manually across irregular schedules."
    ], ACCENT_ROSE)

    add_card(s2, Inches(6.8), Inches(1.6), Inches(5.7), Inches(2.5), "The Solution: Hybrid AML & DL Telemetry", [
        "Curated 2.82M sensor observations (1940-2023) across England, USA, Ireland, China, and Canada.",
        "Engineered 38 features with log1p skewness taming, chronological station lags, and guideline stress flags.",
        "Deployed both tree ensembles (Random Forest, XGBoost) and a Multi-Task Dual-Head PyTorch MLP."
    ], ACCENT_CYAN)

    add_card(s2, Inches(0.8), Inches(4.3), Inches(11.733), Inches(2.5), "Key Achievements & Breakthroughs", [
        "Data Integrity: Disproved the '35% missing data' myth; proved 100% complete case ground truth (0 nulls) across 14 columns.",
        "Domain Physical Cleaning: Removed only 8,360 rows (0.296%) with physically impossible sensor values (negative concentrations, abnormal pH).",
        "State-of-the-Art Accuracy: Random Forest attained R² = 0.9999; PyTorch Dual-Head MLP achieved 96.09% accuracy and 97.49% recall on critical 'Poor' water quality tier.",
        "Production Version Control: Resolved GitHub's 100 MB per-file limit using transparent 3-part Parquet sharding (~53 MB each)."
    ], ACCENT_EMERALD)

    # ==========================================
    # SLIDE 3: Understanding the Target: CCME WQI
    # ==========================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Understanding the Target: Canadian Council of Ministers WQI", slide_num=3)

    add_card(s3, Inches(0.8), Inches(1.6), Inches(6.0), Inches(5.2), "Mathematical Formulation", [
        "The Canadian Council of Ministers of the Environment (CCME) WQI combines three distinct compliance factors:",
        "CCME WQI = 100 - [ sqrt(F1² + F2² + F3²) / 1.732 ]",
        "• F1 (Scope): % of chemical variables failing to meet objectives at least once during surveillance period.",
        "• F2 (Frequency): % of individual analytical measurements that violate permissible limits.",
        "• F3 (Amplitude): Magnitude of exceedances, computed via normalized sum of excursions (nse).",
        "Scalar output spans 0 (severely degraded) to 100 (pristine natural quality).",
        "Piecewise non-linear formulation creates sharp regression thresholds ideal for tree and deep learning models."
    ], ACCENT_BLUE)

    # Quality Tiers on Right
    add_card(s3, Inches(7.1), Inches(1.6), Inches(5.4), Inches(0.95), "Excellent: 95 – 100", [
        "Water quality is protected with virtual absence of threat or impairment."
    ], ACCENT_CYAN)
    add_card(s3, Inches(7.1), Inches(2.65), Inches(5.4), Inches(0.95), "Good: 80 – 94", [
        "Conditions rarely depart from natural or desirable levels."
    ], ACCENT_EMERALD)
    add_card(s3, Inches(7.1), Inches(3.7), Inches(5.4), Inches(0.95), "Fair: 65 – 79", [
        "Conditions sometimes depart from natural levels; quality occasionally compromised."
    ], ACCENT_BLUE)
    add_card(s3, Inches(7.1), Inches(4.75), Inches(5.4), Inches(0.95), "Marginal: 45 – 64", [
        "Water quality is frequently impaired; conditions often depart from desirable levels."
    ], ACCENT_AMBER)
    add_card(s3, Inches(7.1), Inches(5.8), Inches(5.4), Inches(0.95), "Poor: 0 – 44", [
        "Water quality is almost always impaired; poses chronic threat to aquatic life."
    ], ACCENT_ROSE)

    # ==========================================
    # SLIDE 4: Data Provenance & Cleaning Triage
    # ==========================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Data Provenance & Cleaning Triage: Establishing Truth", slide_num=4)

    headers_s4 = ["Milestone", "Previous Misconception", "Empirical Ground Truth", "Action Taken"]
    rows_s4 = [
        ["Geographic Scope", "Unchecked inclusion of India & Australia scrapers", "5 authentic validated nations: England, USA, Ireland, China, Canada", "Purged unaligned scrapers; rebuilt dataset from raw verified files"],
        ["Missing Data Audit", "Belief that dataset had ~35% missing values", "0 null values across all 2,827,977 rows and 14 columns", "Implemented 100% Complete Case Analysis; rejected synthetic imputation"],
        ["Sensor Calibration", "Raw telemetry assumed physically valid", "Sensors exhibit zero-drift and negative values (BOD < 0, DO < 0)", "Enforced aquatic chemistry domain constraints on all chemical sensors"],
        ["Outlier Handling", "Arbitrary statistical IQR clipping", "IQR clipping destroys true pollution shock spikes", "Retained real chemical spikes; only purged non-physical sensor faults (0.296%)"]
    ]
    add_table_data(s4, Inches(0.8), Inches(1.6), Inches(11.733), Inches(3.6), headers_s4, rows_s4, [Inches(2.0), Inches(3.0), Inches(3.2), Inches(3.533)])

    add_card(s4, Inches(0.8), Inches(5.4), Inches(11.733), Inches(1.5), "Scientific Takeaway on Complete Case Analysis", [
        "Why no KNN or MICE imputation? Imputing 35% synthetic values would distort delicate stoichiometric nutrient ratios (N:P) and artificial guidelines.",
        "Confirming 0 nulls ground truth preserved complete chemical fidelity for all 2,819,617 training and validation records."
    ], ACCENT_CYAN)

    # ==========================================
    # SLIDE 5: Domain Physical Outlier Filtering
    # ==========================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Domain Physical Outlier Filtering: Aquatic Chemistry Bounds", slide_num=5)

    headers_s5 = ["Parameter", "Sensor Unit", "Valid Physical Range", "Violations Found", "Chemical Rationale"]
    rows_s5 = [
        ["pH", "pH Units", "3.0 – 12.0", "345 rows", "Acid rain limit is ~pH 3.0; caustic industrial bleach exceeds pH 12.0"],
        ["Temperature", "°C", "0.0 – 45.0", "128 rows", "Water freezes below 0°C; natural surface ambient rarely exceeds 45°C"],
        ["Dissolved Oxygen", "mg/L", "0.0 – 25.0", "892 rows", "Negative DO is impossible; 25 mg/L is super-saturation physical limit"],
        ["Conductivity", "µS/cm", "≥ 0.0", "512 rows", "Conductance represents ionic concentration and cannot be negative"],
        ["BOD5", "mg/L", "≥ 0.0", "1,240 rows", "Microbial oxygen demand cannot be negative"],
        ["Nutrients (N, P, NH4)", "mg/L", "≥ 0.0", "5,243 rows", "Negative mass concentration indicates instrument zero-point calibration drift"]
    ]
    add_table_data(s5, Inches(0.8), Inches(1.6), Inches(7.5), Inches(4.2), headers_s5, rows_s5, [Inches(1.5), Inches(1.0), Inches(1.4), Inches(1.1), Inches(2.5)])

    add_metric_card(s5, Inches(8.6), Inches(1.6), Inches(3.9), Inches(1.3), "8,360", "Rows Removed", "0.296% of dataset (Instrument Faults)", ACCENT_ROSE)
    add_metric_card(s5, Inches(8.6), Inches(3.05), Inches(3.9), Inches(1.3), "2,819,617", "Preserved Rows", "99.704% retention of genuine data", ACCENT_EMERALD)
    add_card(s5, Inches(8.6), Inches(4.5), Inches(3.9), Inches(2.3), "Engineering Principle", [
        "Statistical IQR clipping is destructive for environmental modeling because true industrial pollution events manifest as extreme upper outliers.",
        "We strictly removed only physically impossible telemetry errors."
    ], ACCENT_BLUE)

    # ==========================================
    # SLIDE 6: Global Landscape: Geography & Waterbodies
    # ==========================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Global Landscape: Geographic & Waterbody Distributions", slide_num=6)

    fig1 = get_fig("data/GeneratedOutput/01.EDA/figures/01_country_distribution.png")
    fig2 = get_fig("data/GeneratedOutput/01.EDA/figures/02_waterbody_distribution.png")

    if fig1 and fig2:
        s6.shapes.add_picture(fig1, Inches(0.8), Inches(1.6), width=Inches(5.7))
        s6.shapes.add_picture(fig2, Inches(6.8), Inches(1.6), width=Inches(5.7))
    elif fig1:
        s6.shapes.add_picture(fig1, Inches(0.8), Inches(1.6), width=Inches(6.5))

    add_card(s6, Inches(0.8), Inches(5.35), Inches(5.7), Inches(1.7), "Geographical Representation", [
        "England represents 75.5% of surveillance stations, reflecting the UK Environment Agency's dense monitoring grid.",
        "USA (14.4%), Ireland (8.3%), China (1.6%), and Canada (0.1%) provide multi-continental hydrological diversity."
    ], ACCENT_CYAN)

    add_card(s6, Inches(6.8), Inches(5.35), Inches(5.7), Inches(1.7), "12 Waterbody Classes", [
        "Rivers dominate the dataset (65.3%), followed by Canals (10.1%) and Lakes (8.2%).",
        "Estuarine, coastal, and groundwater bodies capture brackish salinity transitions and distinct ecological baselines."
    ], ACCENT_BLUE)

    # ==========================================
    # SLIDE 7: 83 Years of Water Monitoring
    # ==========================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "83 Years of Environmental Surveillance: 1940 to 2023", slide_num=7)

    fig3 = get_fig("data/GeneratedOutput/01.EDA/figures/03_temporal_coverage.png")
    if fig3:
        s7.shapes.add_picture(fig3, Inches(0.8), Inches(1.6), width=Inches(7.2))

    add_card(s7, Inches(8.3), Inches(1.6), Inches(4.2), Inches(2.6), "Historical Surveillance Dynamics", [
        "Dataset spans 83 years of environmental monitoring across international hydrological basins.",
        "Pre-1990: Sparse baseline sampling focused on acute municipal sanitary concerns and industrial outflows.",
        "Post-2000: Massive exponential acceleration driven by the EU Water Framework Directive and modern sensor telemetry."
    ], ACCENT_AMBER)

    add_card(s7, Inches(8.3), Inches(4.4), Inches(4.2), Inches(2.5), "Temporal Feature Engineering", [
        "Continuous monitoring requires explicit cyclical time representations.",
        "Engineered Sine/Cosine harmonic transforms on month and day-of-year: month_sin, month_cos.",
        "Enables ML models to capture summer eutrophication and winter dissolved oxygen surges seamlessly."
    ], ACCENT_CYAN)

    # ==========================================
    # SLIDE 8: Sensor Distributions & Geochemistry
    # ==========================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Sensor Distributions & Geochemical Thermodynamics", slide_num=8)

    fig7 = get_fig("data/GeneratedOutput/01.EDA/figures/07_do_temperature_saturation.png")
    fig9 = get_fig("data/GeneratedOutput/01.EDA/figures/09_np_ratio.png")

    if fig7 and fig9:
        s8.shapes.add_picture(fig7, Inches(0.8), Inches(1.6), width=Inches(5.7))
        s8.shapes.add_picture(fig9, Inches(6.8), Inches(1.6), width=Inches(5.7))

    add_card(s8, Inches(0.8), Inches(5.35), Inches(5.7), Inches(1.7), "Henry's Law & Dissolved Oxygen", [
        "DO vs. Temperature exhibits negative non-linear curvature obeying Benson & Krause solubility equations.",
        "Thermal pollution directly suppresses dissolved oxygen capacity, triggering hypoxic stress in summer."
    ], ACCENT_CYAN)

    add_card(s8, Inches(6.8), Inches(5.35), Inches(5.7), Inches(1.7), "Redfield Stoichiometry (N:P Ratio)", [
        "92.4% of samples exhibit N:P ratios well above the classical 16:1 Redfield oceanic baseline.",
        "Confirms that phosphorus (Orthophosphate) is the critical limiting nutrient driving freshwater eutrophication."
    ], ACCENT_EMERALD)

    # ==========================================
    # SLIDE 9: Feature Engineering: Taming Skewness
    # ==========================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Feature Engineering: Taming Skewness via Log1p Transforms", slide_num=9)

    headers_s9 = ["Chemical Parameter", "Raw Skewness", "Log1p Skewness", "Raw Correlation (r)", "Log1p Correlation (r)", "Predictive Gain"]
    rows_s9 = [
        ["Orthophosphate (P)", "+14.82", "+0.91", "-0.6582", "-0.8491", "+29.0% Linear Gain"],
        ["Total Oxidised Nitrogen (N)", "+6.41", "+0.23", "-0.3120", "-0.4812", "+54.2% Linear Gain"],
        ["Ammoniacal Nitrogen (NH4)", "+21.15", "+1.14", "-0.2450", "-0.4385", "+79.0% Linear Gain"],
        ["Biochemical Oxygen Demand", "+18.90", "+1.05", "-0.3812", "-0.5120", "+34.3% Linear Gain"],
        ["N:P Stoichiometry Ratio", "+32.40", "+0.45", "+0.0120", "+0.5210", "Order of Magnitude Boost"]
    ]
    add_table_data(s9, Inches(0.8), Inches(1.6), Inches(11.733), Inches(3.2), headers_s9, rows_s9, [Inches(2.5), Inches(1.5), Inches(1.5), Inches(1.8), Inches(1.8), Inches(2.633)])

    add_card(s9, Inches(0.8), Inches(5.0), Inches(5.7), Inches(2.0), "Why Log1p Was Essential", [
        "Environmental pollutants span 4 to 5 orders of magnitude due to industrial discharge pulses.",
        "Un-normalized distributions cause gradient explosion in neural networks and sub-optimal splitting in tree regressors.",
        "Applying y = log(1 + x) compresses extreme tails while preserving zero baseline."
    ], ACCENT_BLUE)

    add_card(s9, Inches(6.8), Inches(5.0), Inches(5.7), Inches(2.0), "The Orthophosphate Breakthrough", [
        "Orthophosphate correlation with CCME jumped from -0.66 to -0.85 after log1p transformation.",
        "This single feature explains nearly half of the variance in global water quality indices."
    ], ACCENT_EMERALD)

    # ==========================================
    # SLIDE 10: Feature Engineering: Station History
    # ==========================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Feature Engineering: Chronological Station Environmental Memory", slide_num=10)

    add_card(s10, Inches(0.8), Inches(1.6), Inches(6.0), Inches(3.2), "Capturing Environmental Inertia", [
        "A chemical reading taken in isolation ignores the river basin's recent ecological history.",
        "Rivers, aquifers, and lakes possess inertia: a river suffering from eutrophication yesterday will likely still be polluted today.",
        "Grouped all observations by unique monitoring station (Area) and sorted chronologically.",
        "Created lag-1 and rolling-3 statistics (mean, std) for key parameters: DO, Orthophosphate, Temperature."
    ], ACCENT_CYAN)

    add_card(s10, Inches(7.1), Inches(1.6), Inches(5.4), Inches(3.2), "Strict Leakage Prevention Protocol", [
        "Crucial Rule: NEVER use future records to predict current state.",
        "Lagged features were computed strictly using pandas .shift(1) within station groups.",
        "Missing historical values for the first station observation were imputed with historical global medians.",
        "Result: Prior station orthophosphate alone exhibits rho = -0.78 Spearman rank correlation with current CCME WQI!"
    ], ACCENT_AMBER)

    # Bottom 3 cards highlighting station memory metrics
    add_metric_card(s10, Inches(0.8), Inches(5.0), Inches(3.7), Inches(1.9), "rho = -0.78", "Prior Station Ortho", "Spearman rank correlation of hist_ortho_lag1 with current WQI", ACCENT_EMERALD)
    add_metric_card(s10, Inches(4.8), Inches(5.0), Inches(3.7), Inches(1.9), "rho = +0.64", "Prior Station DO", "Spearman rank correlation of hist_do_lag1 with current WQI", ACCENT_BLUE)
    add_metric_card(s10, Inches(8.8), Inches(5.0), Inches(3.7), Inches(1.9), "6 Features", "Historical Memory Set", "Lag1 & rolling-3 stats for DO, Ortho, and Water Temperature", ACCENT_CYAN)

    # ==========================================
    # SLIDE 11: System Architecture & Leakage Prevention
    # ==========================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "End-to-End Pipeline & Zero-Leakage Architecture", slide_num=11)

    add_card(s11, Inches(0.8), Inches(1.6), Inches(3.7), Inches(5.3), "1. Dataset Partitioning", [
        "Total Clean Rows: 2,819,617",
        "Split Strategy: 70 / 15 / 15 stratified random split on WQI class.",
        "• Train Set: 1,973,731 rows (70%)",
        "• Val Set: 422,943 rows (15%)",
        "• Test Set: 422,943 rows (15%)",
        "Stratification guarantees identical representation of rare 'Poor' and 'Marginal' classes across all splits."
    ], ACCENT_BLUE)

    add_card(s11, Inches(4.8), Inches(1.6), Inches(3.7), Inches(5.3), "2. Target Quarantine", [
        "CCME WQI is mathematically calculated from sensor values.",
        "Strict Quarantine Protocol:",
        "• Raw ccme_values, ccme_wqi, and ccme_wqi_code strictly isolated.",
        "• Hard programmatic assertions verify zero target-derived columns exist in feature matrix X.",
        "StandardScaler fitted strictly on training partition only; applied to validation and test partitions."
    ], ACCENT_ROSE)

    add_card(s11, Inches(8.8), Inches(1.6), Inches(3.7), Inches(5.3), "3. Input Feature Matrix (38)", [
        "• 8 Base Chemical Sensors",
        "• 7 Stoichiometric & Toxicity Ratios",
        "• 7 Log1p Transformed Nutrients",
        "• 1 Multi-Guideline Stress Counter",
        "• 6 Chronological Station Lags",
        "• 2 Contextual (Waterbody, Country)",
        "• 7 Temporal & Cyclical Harmonics",
        "Zero missing values; fully scaled."
    ], ACCENT_EMERALD)

    # ==========================================
    # SLIDE 12: Phase 1: Machine Learning Tree Models
    # ==========================================
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "Phase 1: Machine Learning Tree Ensembles (RF & XGBoost)", slide_num=12)

    fig16 = get_fig("data/GeneratedOutput/04.ModelEvaluation/model_comparison.png")
    if fig16:
        s12.shapes.add_picture(fig16, Inches(0.8), Inches(1.6), width=Inches(6.8))

    add_card(s12, Inches(7.9), Inches(1.6), Inches(4.6), Inches(2.7), "Model Hyperparameter Tuning", [
        "Random Forest Regressor:",
        "• 30 Estimators, Max Depth = 14",
        "• Min Samples Leaf = 5, n_jobs = -1",
        "• Val MAE = 0.0521 | Val R² = 0.9999",
        "XGBoost Regressor:",
        "• 150 Estimators, Max Depth = 8, LR = 0.1",
        "• tree_method = 'hist' (histogram binning)",
        "• Val MAE = 0.1375 | Val R² = 0.9996"
    ], ACCENT_CYAN)

    add_card(s12, Inches(7.9), Inches(4.5), Inches(4.6), Inches(2.4), "Why Tree Ensembles Excel Here", [
        "CCME WQI uses piecewise step thresholds (F1, F2 exceedances).",
        "Orthogonal axis-aligned decision tree splits match guideline boundaries with near-perfect mathematical precision.",
        "Both models outperform the Mean Baseline by over 99.6% error reduction."
    ], ACCENT_EMERALD)

    # ==========================================
    # SLIDE 13: Phase 2: Deep Learning Dual-Head MLP
    # ==========================================
    s13 = prs.slides.add_slide(blank_layout)
    add_header(s13, "Phase 2: Deep Learning Dual-Head Multi-Task Neural Network", slide_num=13)

    add_card(s13, Inches(0.8), Inches(1.6), Inches(5.7), Inches(5.3), "Dual-Head PyTorch Architecture", [
        "Input Layer: 38 Scaled Geochemical Features",
        "Shared Representation Trunk:",
        "• Dense 256 -> BatchNorm -> GELU -> Dropout(0.2)",
        "• Dense 128 -> BatchNorm -> GELU -> Dropout(0.15)",
        "• Dense 64  -> BatchNorm -> GELU -> Dropout(0.1)",
        "Head A: Continuous Regression Head",
        "• Dense 32 -> GELU -> Dense 1 -> Sigmoid * 100",
        "• Predicts continuous CCME scalar index (0 - 100)",
        "Head B: Discrete Classification Head",
        "• Dense 32 -> GELU -> Dense 5 (Logits)",
        "• Predicts 5 WQI regulatory classes simultaneously."
    ], ACCENT_BLUE)

    add_card(s13, Inches(6.8), Inches(1.6), Inches(5.7), Inches(2.6), "Multi-Task Joint Loss Formulation", [
        "Loss_total = Loss_reg + 0.5 * Loss_cls",
        "• Loss_reg: Smooth L1 Loss (Huber) for outlier robustness.",
        "• Loss_cls: Cross-Entropy Loss with Inverse Class Frequency Weights:",
        "  w_c = N_total / (5 * N_c)",
        "Prevents the dominant 'Good' and 'Fair' classes from drowning out critical 'Poor' class detections."
    ], ACCENT_CYAN)

    add_card(s13, Inches(6.8), Inches(4.4), Inches(5.7), Inches(2.5), "Hardware Acceleration on Apple Silicon", [
        "PyTorch training leveraged torch.device('mps') on Apple Silicon M-series GPU.",
        "High-throughput batch size of 2,048 samples.",
        "AdamW Optimizer (lr = 1e-3, weight_decay = 1e-4) with ReduceLROnPlateau scheduler.",
        "Trained across 1.97M rows in 8 rapid epochs with early stopping."
    ], ACCENT_EMERALD)

    # ==========================================
    # SLIDE 14: Deep Learning Performance & Minority Recall
    # ==========================================
    s14 = prs.slides.add_slide(blank_layout)
    add_header(s14, "Deep Learning Performance: 96.09% Accuracy & Minority Recall", slide_num=14)

    fig14 = get_fig("data/GeneratedOutput/03.ModelTraining/dl_training_curves.png")
    fig15 = get_fig("data/GeneratedOutput/03.ModelTraining/dl_confusion_matrix.png")

    if fig14 and fig15:
        s14.shapes.add_picture(fig14, Inches(0.8), Inches(1.6), width=Inches(5.7))
        s14.shapes.add_picture(fig15, Inches(6.8), Inches(1.6), width=Inches(5.7))

    add_card(s14, Inches(0.8), Inches(5.35), Inches(5.7), Inches(1.7), "Training Loss Convergence", [
        "Validation loss tracked training loss smoothly with zero divergence or overfitting.",
        "Regression head converged to Val R² = 0.9957 and Val MAE = 0.7056 index points."
    ], ACCENT_CYAN)

    add_card(s14, Inches(6.8), Inches(5.35), Inches(5.7), Inches(1.7), "97.49% Recall & Calibrated Uncertainty", [
        "Inverse class weighting successfully resolved extreme class imbalance (97.49% Poor recall).",
        "Probability Calibration: Multi-Class Brier Score = 0.0533 | ECE = 1.27% across all 10 confidence deciles."
    ], ACCENT_ROSE)

    # ==========================================
    # SLIDE 15: The AML vs. DL Showdown
    # ==========================================
    s15 = prs.slides.add_slide(blank_layout)
    add_header(s15, "The AML vs. DL Showdown: Comprehensive Model Benchmark", slide_num=15)

    headers_s15 = ["Model Name", "Model Family", "Val MAE", "Val RMSE", "Val R²", "Classification Acc", "Inference Latency"]
    rows_s15 = [
        ["Mean Baseline", "Heuristic", "14.1297", "17.6561", "-0.0000", "N/A", "< 0.01 ms"],
        ["Random Forest Regressor", "Bagged Trees (30)", "0.0521", "0.2081", "0.9999", "N/A", "1.24 ms"],
        ["XGBoost Regressor", "Gradient Boosted (150)", "0.1375", "0.3740", "0.9996", "N/A", "0.45 ms"],
        ["Dual-Head MLP (PyTorch)", "Deep Multi-Task Neural Net", "0.7056", "1.1557", "0.9957", "96.09% (ECE 1.27%)", "0.08 ms (Batched GPU)"]
    ]
    add_table_data(s15, Inches(0.8), Inches(1.6), Inches(11.733), Inches(2.8), headers_s15, rows_s15, [Inches(2.5), Inches(2.2), Inches(1.2), Inches(1.2), Inches(1.2), Inches(1.8), Inches(1.633)])

    add_card(s15, Inches(0.8), Inches(4.6), Inches(5.7), Inches(2.3), "When to Use Random Forest / XGBoost", [
        "Unbeatable scalar regression precision (MAE = 0.052 index points).",
        "Ideal for embedded sensor hardware and offline reporting where only continuous scalar CCME score is required.",
        "Native handling of unscaled feature spaces."
    ], ACCENT_EMERALD)

    add_card(s15, Inches(6.8), Inches(4.6), Inches(5.7), Inches(2.3), "When to Use Dual-Head Deep Learning", [
        "Provides simultaneous continuous score + calibrated probability distribution over all 5 quality tiers.",
        "Uncertainty quantification: Softmax entropy highlights borderline readings needing lab verification.",
        "15x faster batched throughput on GPU clusters for national monitoring surveillance."
    ], ACCENT_BLUE)

    # ==========================================
    # SLIDE 16: Explainable AI: Feature Importance & TreeSHAP
    # ==========================================
    s16 = prs.slides.add_slide(blank_layout)
    add_header(s16, "Explainable AI: Feature Importance & TreeSHAP Attribution", slide_num=16)

    fig17 = get_fig("data/GeneratedOutput/04.ModelEvaluation/feature_importance.png")
    fig_shap = get_fig("data/GeneratedOutput/04.ModelEvaluation/shap_summary.png")

    if fig17 and fig_shap:
        s16.shapes.add_picture(fig17, Inches(0.8), Inches(1.6), width=Inches(5.7))
        s16.shapes.add_picture(fig_shap, Inches(6.8), Inches(1.6), width=Inches(5.7))
    elif fig17:
        s16.shapes.add_picture(fig17, Inches(0.8), Inches(1.6), width=Inches(7.2))

    add_card(s16, Inches(0.8), Inches(5.35), Inches(5.7), Inches(1.7), "Global Tree Importance (Gini/Gain)", [
        "Orthophosphate (log1p_orthophosphate + raw) accounts for ~88% of all model split decisions.",
        "Validates limnological science: phosphorus is the limiting nutrient controlling algae blooms."
    ], ACCENT_EMERALD)

    add_card(s16, Inches(6.8), Inches(5.35), Inches(5.7), Inches(1.7), "Local TreeSHAP Beeswarm Attribution", [
        "Game-theoretic SHAP verifies high orthophosphate pushes CCME WQI down sharply.",
        "Consistent attribution across both methods guarantees scientific explainability."
    ], ACCENT_CYAN)

    # ==========================================
    # SLIDE 17: Engineering Triumph: Parquet Sharding & Modular Architecture
    # ==========================================
    s17 = prs.slides.add_slide(blank_layout)
    add_header(s17, "Engineering Triumph: Parquet Sharding, Modular src/ & pytest Suite", slide_num=17)

    add_card(s17, Inches(0.8), Inches(1.6), Inches(5.7), Inches(2.6), "Multi-Part Parquet Sharding (GitHub)", [
        "188 MB training partitions blocked by GitHub 100 MB limit.",
        "Engineered transparent 3-part Parquet sharding (~53 MB each) inside train.parquet/ using zstd compression.",
        "Zero code changes required: pd.read_parquet('train.parquet') loads all parts seamlessly into 1.97M rows."
    ], ACCENT_ROSE)

    add_card(s17, Inches(6.8), Inches(1.6), Inches(5.7), Inches(2.6), "Modular src/ Package & 100% pytest Suite", [
        "Refactored all pipeline logic into a production-grade Python package: src/features, src/models, src/metrics, src/utils.",
        "Comprehensive automated test suite in tests/ (12 unit tests, 100% passing in 2.2s):",
        "• Validates aquatic physical bounds & zero lookahead leakage.",
        "• Validates PyTorch model forward pass, shapes, and target quarantine."
    ], ACCENT_CYAN)


    add_card(s17, Inches(0.8), Inches(4.3), Inches(11.733), Inches(2.6), "Seamless Developer Experience & Code Compatibility", [
        "Zero Code Changes Required for Downstream Pipelines: Apache Arrow and Pandas natively load multi-part directories transparently:",
        "   >>> df_train = pd.read_parquet('data/Dataset/02.Processed/train.parquet')",
        "Automatically reads all 3 sub-parts in parallel into a single unified 1,973,731-row DataFrame!",
        "Successfully pushed 515.44 MB of code, models, data, and figures to GitHub origin/main with zero files ignored."
    ], ACCENT_EMERALD)

    # ==========================================
    # SLIDE 18: Conclusions & Future Directions
    # ==========================================
    s18 = prs.slides.add_slide(blank_layout)
    add_header(s18, "Conclusions, Key Takeaways & Future Directions", slide_num=18)

    add_card(s18, Inches(0.8), Inches(1.6), Inches(5.7), Inches(3.6), "Summary of Achievements", [
        "1. Complete Ground Truth: Debunked the '35% missing data' myth; verified 0 nulls across 2.82M international records.",
        "2. Domain Physical Cleaning: Enforced aquatic chemistry boundaries, safely eliminating sensor drift with 99.7% data retention.",
        "3. Feature Innovations: Log1p transforms and chronological station lags boosted predictive correlations up to +79%.",
        "4. Benchmark Dominance: Random Forest achieved R² = 0.9999; PyTorch Dual-Head MLP achieved 96.09% accuracy and 97.49% recall on 'Poor' water quality tier.",
        "5. Clean Git Workflow: Multi-part Parquet sharding pushed 100% of data to GitHub without Git LFS."
    ], ACCENT_EMERALD)

    add_card(s18, Inches(6.8), Inches(1.6), Inches(5.7), Inches(3.6), "Future Directions & Deployment", [
        "1. Real-Time IoT Edge Telemetry: Deploy quantized ONNX neural network weights directly to river sensor buoys for instant alerts.",
        "2. Spatio-Temporal Graph Neural Networks (GNNs): Model river stream networks as directed flow graphs to track downstream pollution dispersion.",
        "3. Satellite Remote Sensing Integration: Fuse in-situ sensor telemetry with Sentinel-2 multi-spectral chlorophyll-a reflectance imagery.",
        "4. Public Environmental API: Wrap inference engine in FastAPI microservice for municipal environmental agencies."
    ], ACCENT_BLUE)

    add_card(s18, Inches(0.8), Inches(5.4), Inches(11.733), Inches(1.5), "Thank You! Questions & Discussion", [
        "Repository: https://github.com/MAYANKSHARMA01010/Semester_5_AML_DL  |  Author: Mayank Sharma",
        "Complete technical report, notebooks, figures, and model weights are fully documented in the repository."
    ], ACCENT_CYAN)

    prs.save(OUTPUT_PATH)
    print(f"Presentation saved successfully to: {OUTPUT_PATH}")

if __name__ == "__main__":
    create_presentation()
