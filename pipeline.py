"""
pipeline.py
-----------
Main orchestrator for the Data Cleaning Pipeline.

Usage:
    python pipeline.py

Steps:
    1.  Load file (CSV / Excel / JSON) -> UTF-8 in memory
    2.  Lowercase all column names
    3.  Auto-drop constant columns (zero variance - no user input needed)
    4.  Show BEFORE overview (shape, nulls, duplicates)
    5.  Show BEFORE charts (displayed on screen)
    6.  User configuration:
          a. Which high-cardinality columns to DROP (USER chooses - nothing auto-dropped)
          b. Which columns for selective duplicate removal
          c. Which columns for outlier removal
    7.  Run 10 cleaning steps with progress banners
    8.  Validate cleaned dataset
    9.  Show AFTER overview + AFTER charts (displayed on screen)
    10. Save cleaned CSV + HTML report + log -> output/
"""

import os
import sys
import io
import pandas as pd
import numpy as np
from datetime import datetime

# -- Local modules -------------------------------------------------------------
from logger    import setup_logger
from cleaner   import (
    verify_columns,
    lowercase_columns,
    clean_whitespace,
    fix_date_inconsistency,
    drop_constant_columns,
    remove_all_duplicates,
    remove_duplicates_by_list,
    strip_symbols_conditionally,
    fix_majority_type,
    smart_impute,
    fix_typos_with_rapidfuzz,
    remove_outliers,
    validate_cleaned_data,
)
from visualizer import generate_before_charts, generate_after_charts
from reporter   import generate_html_report

# -- Output directory ----------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


# ==============================================================================
# DISPLAY HELPERS
# ==============================================================================

def _separator(char="─", width=65):
    print(char * width)

def _header(text: str):
    _separator("═")
    print(f"  {text}")
    _separator("═")

def _step_banner(step_num: int, total: int, name: str):
    print(f"\n  [{step_num}/{total}] >  {name}")
    _separator(".")

def _done(msg: str = "Step complete."):
    print(f"  OK  {msg}\n")

def _get_df_stats(df: pd.DataFrame) -> dict:
    return {
        "rows":       df.shape[0],
        "cols":       df.shape[1],
        "missing":    int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "memory_kb":  round(df.memory_usage(deep=True).sum() / 1024, 2),
    }

def _print_df_overview(df: pd.DataFrame, label: str = ""):
    print(f"\n  --- {label if label else 'Dataset Overview'} ---")
    _separator()
    print(f"  Shape          : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Duplicates     : {df.duplicated().sum():,} rows")
    print(f"  Missing Values : {df.isnull().sum().sum():,} total")
    print(f"  Memory Usage   : {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    print()
    buf = io.StringIO()
    df.info(buf=buf)
    for line in buf.getvalue().splitlines():
        print(f"  {line}")
    print()
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print("  Columns with missing values:")
        for col, cnt in missing.items():
            pct = cnt / len(df) * 100
            print(f"    - {col:<30} {cnt:>6,} missing  ({pct:.1f}%)")
    else:
        print("  No missing values found.")
    _separator()


# ==============================================================================
# FILE LOADING
# ==============================================================================

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json"}

def load_file(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"\nERROR: File not found: '{file_path}'\n"
            f"  Please check the path and try again."
        )
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"\nERROR: Unsupported format: '{ext}'\n"
            f"  Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    print(f"\n  Loading: {file_path}  (format: {ext})")
    try:
        if ext == ".csv":
            df = pd.read_csv(file_path, encoding="utf-8", low_memory=False)
        elif ext in {".xlsx", ".xls"}:
            df = pd.read_excel(file_path, engine="openpyxl")
            print("  Excel loaded -> converted to in-memory DataFrame (UTF-8).")
        elif ext == ".json":
            df = pd.read_json(file_path, encoding="utf-8")
            print("  JSON loaded -> converted to in-memory DataFrame (UTF-8).")
    except Exception as e:
        raise RuntimeError(f"\nERROR: Failed to read '{file_path}'.\n  {e}")
    print(f"  Loaded successfully. Shape: {df.shape}")
    return df


# ==============================================================================
# COLUMN INPUT HELPER  (re-ask loop on bad names)
# ==============================================================================

def _ask_column_list(df: pd.DataFrame, prompt: str) -> list:
    """
    Asks user for comma-separated column names.
    Loops until all names are valid, or user presses Enter to skip.
    """
    print(f"\n  {prompt}")
    print(f"  Available columns: {list(df.columns)}")
    print("  (Press Enter to skip this step)")

    while True:
        raw = input("  > ").strip()
        if not raw:
            print("  Skipped.")
            return []

        user_list = [c.strip() for c in raw.split(",") if c.strip()]
        verified  = verify_columns(df, user_list)
        invalid   = [c for c in user_list if c not in verified]

        if invalid:
            print(f"\n  WARNING - These column(s) were NOT found: {invalid}")
            print(f"  Valid so far: {verified if verified else 'none'}")
            print("  Please fix the typo(s) and re-enter ALL columns, or press Enter to skip.")
            continue

        return verified


# ==============================================================================
# USER CONFIGURATION
# ==============================================================================

def collect_user_inputs(df: pd.DataFrame) -> dict:
    """
    Collects three things from the user:
      1. Which high-cardinality columns to drop (USER picks - nothing auto-dropped)
      2. Which columns for selective duplicate removal
      3. Which columns for outlier removal
    """
    _header("USER CONFIGURATION  (3 questions)")

    # -- 1. High cardinality columns - USER CHOOSES ----------------------------
    print("""
  === QUESTION 1 of 3: High Cardinality Column Drop ===

  INFO: Columns with too many unique values (IDs, emails, free-text) are
  usually not useful for ML models. But YOU decide what to drop.
  Nothing is removed automatically.

  IMPORTANT: Do NOT drop columns you still need. For example, if patient_id
  is your only patient identifier and you need it - keep it.
  Only drop columns that truly have no analytical value.
  """)

    print(f"  {'Column':<30} {'Unique %':>10}  {'Unique Count':>14}  {'Total Rows':>12}")
    print(f"  {'─' * 72}")
    for col in df.columns:
        pct        = df[col].nunique() / len(df) * 100
        unique_cnt = df[col].nunique()
        flag       = "  <- HIGH" if pct >= 95 else ""
        print(f"  {col:<30} {pct:>9.1f}%  {unique_cnt:>14,}  {len(df):>12,}{flag}")

    cardinality_cols = _ask_column_list(
        df,
        "Enter column name(s) to DROP as high-cardinality (comma-separated):"
    )
    if cardinality_cols:
        print(f"  Will drop: {cardinality_cols}")
    else:
        print("  No high-cardinality columns will be dropped.")

    # Work on df minus the cols the user chose to drop
    # so the next two questions only show surviving columns
    df_preview = df.drop(columns=cardinality_cols, errors="ignore")

    # -- 2. Selective duplicate removal ----------------------------------------
    print("""
  === QUESTION 2 of 3: Selective Duplicate Removal ===

  INFO: The pipeline first removes fully duplicate rows (every column matches).
  Here you can ALSO remove rows that duplicate only one SPECIFIC column.

  RECOMMENDATION: Only pick ID-type columns (patient_id, order_id, user_id).
  Picking 'age' or 'city' will accidentally delete valid rows that share a value.
  """)

    print(f"  {'Column':<30} {'Dup Count':>12}  {'Unique Values':>15}  {'Total Rows':>12}")
    print(f"  {'─' * 74}")
    for col in df_preview.columns:
        dup_cnt    = df_preview.duplicated(subset=[col]).sum()
        unique_cnt = df_preview[col].nunique()
        print(f"  {col:<30} {dup_cnt:>12,}  {unique_cnt:>15,}  {len(df_preview):>12,}")

    selective_dup_cols = _ask_column_list(
        df_preview,
        "Enter column name(s) for selective duplicate removal (comma-separated):"
    )

    # Cross-verification - show sample rows that WILL be deleted
    if selective_dup_cols:
        print("\n  --- Cross-Verification: Sample rows that WILL be removed ---")
        for col in selective_dup_cols:
            mask         = df_preview.duplicated(subset=[col], keep="first")
            rows_to_drop = df_preview[mask]
            print(f"\n  Column '{col}': {len(rows_to_drop):,} row(s) will be removed (first kept).")
            if len(rows_to_drop) > 0:
                print("  Sample (up to 5 rows):")
                print(rows_to_drop.head(5).to_string(index=True))
        print()
        confirm = input("  Confirm selective duplicate removal? (yes / no): ").strip().lower()
        if confirm not in ("yes", "y"):
            print("  Cancelled. No selective duplicate columns selected.")
            selective_dup_cols = []
        else:
            print(f"  Confirmed: {selective_dup_cols}")

    # -- 3. Outlier removal columns --------------------------------------------
    print("""
  === QUESTION 3 of 3: Outlier Removal (IQR Method) ===

  INFO: IQR removes rows where a value falls outside:
        [Q1 - 1.5*IQR  to  Q3 + 1.5*IQR]

  STUDY THE TABLE BELOW CAREFULLY:
    - Look at Min and Max for each column
    - If Max is a real possible value -> do NOT include that column
    - If Max looks like a data entry error (e.g. age = 999) -> include it
    - Do NOT select ID columns or binary (0/1) columns
  """)

    # Build numeric candidate list:
    # 1. Proper numeric columns
    # 2. String columns that become numeric after stripping symbols (e.g. $salary)
    num_cols = df_preview.select_dtypes(include=[np.number]).columns.tolist()
    for col in df_preview.select_dtypes(include=["object"]).columns:
        coerced = pd.to_numeric(
            df_preview[col].astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
            errors="coerce"
        )
        if coerced.notna().sum() / len(df_preview) >= 0.7:
            if col not in num_cols:
                num_cols.append(col)

    if num_cols:
        print(f"  {'Column':<25} {'Min':>10} {'Max':>12} {'Mean':>12} "
              f"{'~Outliers':>11} {'Lower Bound':>13} {'Upper Bound':>13}")
        print(f"  {'─' * 100}")
        for col in num_cols:
            # Get numeric series - coerce if needed
            raw = df_preview[col]
            if not pd.api.types.is_numeric_dtype(raw):
                raw = pd.to_numeric(
                    raw.astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
                    errors="coerce"
                )
            data     = raw.dropna()
            if len(data) == 0:
                continue
            Q1       = data.quantile(0.25)
            Q3       = data.quantile(0.75)
            IQR      = Q3 - Q1
            lower    = Q1 - 1.5 * IQR
            upper    = Q3 + 1.5 * IQR
            outliers = ((data < lower) | (data > upper)).sum()
            print(f"  {col:<25} {data.min():>10.2f} {data.max():>12.2f} "
                  f"{data.mean():>12.2f} {outliers:>11,} {lower:>13.2f} {upper:>13.2f}")
    else:
        print("  No numeric columns found in dataset.")

    outlier_cols = _ask_column_list(
        df_preview,
        "Enter column name(s) for outlier removal (comma-separated):"
    )

    return {
        "cardinality_cols":   cardinality_cols,
        "selective_dup_cols": selective_dup_cols,
        "outlier_cols":       outlier_cols,
    }


# ==============================================================================
# CHART DISPLAY  (shows charts interactively on screen)
# ==============================================================================

def _display_charts(df: pd.DataFrame, label: str):
    """
    Displays charts on screen so user can see the data visually.
    Uses TkAgg on Windows for proper interactive display.
    Charts stay open until user closes them.
    """
    import matplotlib
    # TkAgg works on Windows for interactive display
    try:
        matplotlib.use("TkAgg")
    except Exception:
        matplotlib.use("Qt5Agg")

    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")

    BG     = "#0f1117"
    TEXT   = "#e0e0e0"
    ACCENT = "#4fc3f7"
    GRID   = "#2a2a3d"

    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
        "xtick.color": TEXT, "ytick.color": TEXT,
        "text.color": TEXT, "grid.color": GRID,
        "axes.titlecolor": TEXT,
    })

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    figures_opened = 0

    # -- Missing heatmap -------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(max(10, len(df.columns) * 0.9), 5))
    fig1.canvas.manager.set_window_title(f"{label} - Missing Values Heatmap")
    missing_mask = df.isnull()
    if missing_mask.any().any():
        # Use RGBA image directly — seaborn boolean heatmap can render blank
        # due to colormap normalization collapse when values are all 0 or all 1
        mask_arr = missing_mask.values.astype(float)
        if mask_arr.shape[0] > 500:
            step     = max(1, mask_arr.shape[0] // 500)
            mask_arr = mask_arr[::step]
        rgba = np.zeros((*mask_arr.shape, 4))
        rgba[mask_arr == 0] = [0.118, 0.118, 0.180, 1.0]   # dark = present
        rgba[mask_arr == 1] = [0.937, 0.325, 0.314, 1.0]   # red  = missing
        ax1.imshow(rgba, aspect="auto", interpolation="nearest")
        ax1.set_xticks(range(len(df.columns)))
        ax1.set_xticklabels(df.columns, rotation=45, ha="right",
                             fontsize=9, color=TEXT)
        ax1.set_yticks([])
        # Annotate missing count above each column
        for j, col in enumerate(df.columns):
            cnt = missing_mask[col].sum()
            if cnt > 0:
                ax1.text(j, -0.5, f"{cnt}", ha="center", va="bottom",
                         fontsize=8, color="#ef5350", fontweight="bold")
        from matplotlib.patches import Patch
        ax1.legend(handles=[
            Patch(facecolor="#ef5350", label="Missing"),
            Patch(facecolor="#1e1e2e", label="Present"),
        ], loc="upper right", framealpha=0.3, facecolor=BG, edgecolor=GRID)
        ax1.set_title(
            f"[{label}] Missing Values Heatmap  "
            f"(Red = Missing | Total: {int(missing_mask.values.sum()):,})",
            pad=12, color=TEXT
        )
    else:
        ax1.text(0.5, 0.5, "No Missing Values  OK", ha="center", va="center",
                 fontsize=16, color="#66bb6a", fontweight="bold",
                 transform=ax1.transAxes)
        ax1.set_title(f"[{label}] Missing Values Heatmap", pad=10)
        ax1.set_xticks([]); ax1.set_yticks([])
    plt.tight_layout()
    plt.show(block=False)
    figures_opened += 1

    # -- Numeric distributions -------------------------------------------------
    if num_cols:
        n      = len(num_cols)
        cols_r = min(4, n)
        rows_r = (n + cols_r - 1) // cols_r
        fig2, axes2 = plt.subplots(rows_r, cols_r,
                                    figsize=(cols_r * 4, rows_r * 3.2))
        fig2.canvas.manager.set_window_title(f"{label} - Numeric Distributions")
        axes2 = np.array(axes2).flatten()
        for i, col in enumerate(num_cols):
            axes2[i].hist(df[col].dropna(), bins=30, color=ACCENT,
                          alpha=0.8, edgecolor="#1a1a2e")
            axes2[i].set_title(col)
            axes2[i].grid(True, alpha=0.3)
        for j in range(i + 1, len(axes2)):
            axes2[j].set_visible(False)
        fig2.suptitle(f"[{label}] Numeric Distributions", color=TEXT, fontsize=12)
        plt.tight_layout()
        plt.show(block=False)
        figures_opened += 1

    # -- Boxplots --------------------------------------------------------------
    if num_cols:
        fig3, axes3 = plt.subplots(rows_r, cols_r,
                                    figsize=(cols_r * 4, rows_r * 3.2))
        fig3.canvas.manager.set_window_title(f"{label} - Boxplots")
        axes3 = np.array(axes3).flatten()
        for i, col in enumerate(num_cols):
            axes3[i].boxplot(
                df[col].dropna(), patch_artist=True,
                medianprops=dict(color="#ef5350", linewidth=2),
                boxprops=dict(facecolor="#1a237e", color=ACCENT),
                whiskerprops=dict(color=ACCENT),
                capprops=dict(color=ACCENT),
                flierprops=dict(markerfacecolor="#ef5350", marker="o",
                                markersize=4, alpha=0.6)
            )
            axes3[i].set_title(col)
            axes3[i].grid(True, alpha=0.3, axis="y")
        for j in range(i + 1, len(axes3)):
            axes3[j].set_visible(False)
        fig3.suptitle(f"[{label}] Boxplots — Dots beyond whiskers are outliers",
                      color=TEXT, fontsize=12)
        plt.tight_layout()
        plt.show(block=False)
        figures_opened += 1

    # -- Categorical counts ----------------------------------------------------
    if cat_cols:
        n      = len(cat_cols)
        cols_r = min(3, n)
        rows_r = (n + cols_r - 1) // cols_r
        fig4, axes4 = plt.subplots(rows_r, cols_r,
                                    figsize=(cols_r * 5, rows_r * 3.5))
        fig4.canvas.manager.set_window_title(f"{label} - Categorical Value Counts")
        axes4 = np.array(axes4).flatten()
        colors = plt.cm.get_cmap("cool")(np.linspace(0.3, 0.9, 10))
        for i, col in enumerate(cat_cols):
            counts = df[col].value_counts().head(10)
            axes4[i].bar(range(len(counts)), counts.values,
                         color=colors[:len(counts)], alpha=0.85,
                         edgecolor="#1a1a2e")
            axes4[i].set_xticks(range(len(counts)))
            axes4[i].set_xticklabels(counts.index, rotation=35,
                                      ha="right", fontsize=8)
            axes4[i].set_title(col)
            axes4[i].grid(True, alpha=0.3, axis="y")
        for j in range(i + 1, len(axes4)):
            axes4[j].set_visible(False)
        fig4.suptitle(f"[{label}] Categorical Value Counts (top 10 per column)",
                      color=TEXT, fontsize=12)
        plt.tight_layout()
        plt.show(block=False)
        figures_opened += 1

    if figures_opened > 0:
        print(f"  {figures_opened} chart window(s) opened.")
        print(f"  Review them, then CLOSE ALL CHART WINDOWS to continue the pipeline.\n")
        plt.show(block=True)   # blocks here until user closes all windows
    else:
        print("  No columns available to chart.")


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def run_pipeline():
    """Entry point. Runs the full data cleaning pipeline."""

    _header("DATA CLEANING PIPELINE  — Starting Up")

    # -- Get file path ---------------------------------------------------------
    print("\n  Supported formats: CSV (.csv), Excel (.xlsx / .xls), JSON (.json)")
    file_path = input("  Enter the path to your dataset: ").strip().strip('"').strip("'")

    # -- Set up logger ---------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log, log_file = setup_logger(OUTPUT_DIR)
    log.info(f"Pipeline started. Input file: {file_path}")

    # -- Load file -------------------------------------------------------------
    try:
        df = load_file(file_path)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\n{e}")
        sys.exit(1)

    log.info(f"File loaded. Shape: {df.shape}")

    # -- Lowercase column names immediately ------------------------------------
    df = lowercase_columns(df)
    log.info("Column names lowercased.")

    # -- Auto-drop constant columns (no user input needed) ---------------------
    print("\n  Auto-dropping constant columns (zero variance - no useful info)...")
    df = drop_constant_columns(df)
    log.info(f"Constant columns dropped. Shape now: {df.shape}")

    # -- BEFORE overview -------------------------------------------------------
    _header("DATASET OVERVIEW  — Before Cleaning")
    _print_df_overview(df, label="Before Cleaning")
    stats_before = _get_df_stats(df)

    # -- BEFORE charts on screen -----------------------------------------------
    _header("BEFORE-CLEANING VISUALIZATIONS")
    print("""
  Charts are about to open in separate windows:
    - Missing values heatmap  (red = missing cells)
    - Numeric distributions   (histogram per numeric column)
    - Boxplots                (dots beyond whiskers = outliers)
    - Categorical counts      (most frequent values per text column)

  Look at them carefully BEFORE answering the configuration questions.
  Close all chart windows when you are ready to continue.
  """)
    # Generate base64 charts for HTML report FIRST (uses Agg backend silently)
    # Must happen before _display_charts which switches backend to TkAgg
    charts_before = generate_before_charts(df)
    log.info("Before-cleaning charts generated.")

    # Now display charts interactively on screen
    _display_charts(df, "Before")

    # -- User configuration ----------------------------------------------------
    user_cfg           = collect_user_inputs(df)
    cardinality_cols   = user_cfg["cardinality_cols"]
    selective_dup_cols = user_cfg["selective_dup_cols"]
    outlier_cols       = user_cfg["outlier_cols"]

    log.info(f"User config - cardinality_cols:   {cardinality_cols}")
    log.info(f"User config - selective_dup_cols: {selective_dup_cols}")
    log.info(f"User config - outlier_cols:       {outlier_cols}")

    # -- Deep copy for pipeline ------------------------------------------------
    df_clean  = df.copy()
    steps_log = []
    TOTAL     = 10
    _header("RUNNING CLEANING PIPELINE")

    # -- Step 1: Lowercase (already done - confirm) ----------------------------
    _step_banner(1, TOTAL, "Lowercase Column Names")
    df_clean = lowercase_columns(df_clean)
    steps_log.append("Column names lowercased.")
    _done("Column names confirmed lowercase.")

    # -- Step 2: Clean whitespace ----------------------------------------------
    _step_banner(2, TOTAL, "Clean Whitespace in String Columns")
    df_clean = clean_whitespace(df_clean)
    steps_log.append("Whitespace stripped and collapsed in all string columns.")
    _done("Whitespace cleaned.")

    # -- Step 3: Fix date columns ----------------------------------------------
    _step_banner(3, TOTAL, "Standardize Date Columns")
    df_clean = fix_date_inconsistency(df_clean)
    steps_log.append("Date columns standardized.")
    _done("Date columns standardized.")

    # -- Step 4: Drop user-chosen high-cardinality columns ---------------------
    _step_banner(4, TOTAL, "Drop User-Selected High-Cardinality Columns")
    if cardinality_cols:
        valid_card = [c for c in cardinality_cols if c in df_clean.columns]
        df_clean   = df_clean.drop(columns=valid_card)
        steps_log.append(f"Dropped user-selected high-cardinality column(s): {valid_card}")
        log.info(f"Dropped: {valid_card}")
        _done(f"Dropped {len(valid_card)} column(s). Columns now: {df_clean.shape[1]}")
    else:
        steps_log.append("No high-cardinality columns selected for dropping.")
        _done("No columns to drop. Skipped.")

    # -- Step 5: Remove all full duplicates ------------------------------------
    _step_banner(5, TOTAL, "Remove All Full Duplicate Rows")
    before_rows = len(df_clean)
    df_clean    = remove_all_duplicates(df_clean)
    removed     = before_rows - len(df_clean)
    steps_log.append(f"Removed {removed:,} fully duplicate row(s). Rows now: {len(df_clean):,}")
    _done(f"Full duplicates removed. Rows now: {len(df_clean):,}")

    # -- Step 6: Selective duplicate removal -----------------------------------
    _step_banner(6, TOTAL, "Selective Duplicate Removal")
    valid_sel   = [c for c in selective_dup_cols if c in df_clean.columns]
    skipped_sel = [c for c in selective_dup_cols if c not in df_clean.columns]
    if skipped_sel:
        print(f"  WARNING: Column(s) {skipped_sel} no longer exist. Skipping.")
        log.warning(f"Selective dup cols skipped: {skipped_sel}")
    before_rows = len(df_clean)
    df_clean    = remove_duplicates_by_list(df_clean, valid_sel)
    removed     = before_rows - len(df_clean)
    steps_log.append(
        f"Selective dup removal on {valid_sel}: "
        f"{removed:,} row(s) removed. Rows now: {len(df_clean):,}"
    )
    _done(f"Selective duplicates removed. Rows now: {len(df_clean):,}")

    # -- Step 7: Strip symbols conditionally -----------------------------------
    _step_banner(7, TOTAL, "Strip Non-Numeric Symbols from Numeric-Like Columns")
    df_clean = strip_symbols_conditionally(df_clean)
    steps_log.append("Symbol stripping applied to columns with >=70% numeric-like values.")
    _done("Symbol stripping complete.")

    # -- Step 8: Fix majority type ---------------------------------------------
    _step_banner(8, TOTAL, "Enforce Majority Data Type Per Column")
    df_clean = fix_majority_type(df_clean)
    steps_log.append("Majority data type enforced per column.")
    _done("Data types corrected.")

    # -- Step 9: Smart imputation ----------------------------------------------
    _step_banner(9, TOTAL, "Smart Imputation  (Fill Missing Values)")
    missing_before = df_clean.isnull().sum().sum()
    df_clean       = smart_impute(df_clean)
    missing_after  = df_clean.isnull().sum().sum()
    filled         = missing_before - missing_after
    steps_log.append(
        f"Smart imputation: {filled:,} value(s) filled. "
        f"Numeric -> median/mean. Categorical -> mode."
    )
    _done(f"Missing values filled: {filled:,}. Nulls remaining: {missing_after:,}")

    # -- Step 10: Fix typos + Remove outliers ----------------------------------
    _step_banner(10, TOTAL, "Fix Typos  +  Remove Outliers")
    df_clean = fix_typos_with_rapidfuzz(df_clean, threshold=90)
    steps_log.append("Typo fixing applied via RapidFuzz (threshold=90).")

    valid_out   = [c for c in outlier_cols if c in df_clean.columns]
    skipped_out = [c for c in outlier_cols if c not in df_clean.columns]
    if skipped_out:
        print(f"  WARNING: Outlier column(s) {skipped_out} no longer exist. Skipping.")
        log.warning(f"Outlier cols skipped: {skipped_out}")
    before_rows = len(df_clean)
    df_clean    = remove_outliers(df_clean, valid_out)
    removed     = before_rows - len(df_clean)
    steps_log.append(
        f"IQR outlier removal on {valid_out}: "
        f"{removed:,} row(s) removed. Rows now: {len(df_clean):,}"
    )
    _done(f"Typos fixed. Outliers removed: {removed:,}. Rows now: {len(df_clean):,}")

    # -- Final dedup pass ------------------------------------------------------
    # Imputation / column drops can create new identical rows - catch them.
    before_final  = len(df_clean)
    df_clean      = df_clean.drop_duplicates()
    final_removed = before_final - len(df_clean)
    if final_removed > 0:
        print(f"  Final dedup pass removed {final_removed:,} newly created duplicate(s).")
        log.info(f"Final dedup: {final_removed} removed.")
        steps_log.append(f"Final dedup pass: {final_removed:,} duplicate(s) removed.")

    # -- Data validation -------------------------------------------------------
    _header("DATA VALIDATION")
    validation = validate_cleaned_data(df_clean)
    steps_log.append(
        f"Validation: {'PASSED' if validation['passed'] else 'WARNINGS'}. "
        f"Nulls={validation['null_count']}, Duplicates={validation['duplicate_count']}"
    )

    stats_after = _get_df_stats(df_clean)

    # -- AFTER overview --------------------------------------------------------
    _header("DATASET OVERVIEW  — After Cleaning")
    _print_df_overview(df_clean, label="After Cleaning")

    # -- AFTER charts on screen ------------------------------------------------
    _header("AFTER-CLEANING VISUALIZATIONS")
    print("  After-cleaning charts opening — compare with what you saw before.\n")
    _display_charts(df_clean, "After")

    charts_after = generate_after_charts(
        df_before      = df,
        df_after       = df_clean,
        dup_before     = stats_before["duplicates"],
        dup_after      = stats_after["duplicates"],
        missing_before = stats_before["missing"],
        missing_after  = stats_after["missing"],
    )

    # -- Terminal summary ------------------------------------------------------
    _header("PIPELINE SUMMARY")
    print(f"""
  {'Metric':<25} {'Before':>12} {'After':>12} {'Change':>12}
  {'─' * 62}
  {'Rows':<25} {stats_before['rows']:>12,} {stats_after['rows']:>12,} {stats_after['rows'] - stats_before['rows']:>+12,}
  {'Columns':<25} {stats_before['cols']:>12,} {stats_after['cols']:>12,} {stats_after['cols'] - stats_before['cols']:>+12,}
  {'Missing Values':<25} {stats_before['missing']:>12,} {stats_after['missing']:>12,} {stats_after['missing'] - stats_before['missing']:>+12,}
  {'Duplicate Rows':<25} {stats_before['duplicates']:>12,} {stats_after['duplicates']:>12,} {stats_after['duplicates'] - stats_before['duplicates']:>+12,}
  {'Memory (KB)':<25} {stats_before['memory_kb']:>12.2f} {stats_after['memory_kb']:>12.2f} {stats_after['memory_kb'] - stats_before['memory_kb']:>+12.2f}
    """)

    # -- Save cleaned CSV ------------------------------------------------------
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    cleaned_csv = os.path.join(OUTPUT_DIR, f"cleaned_data_{ts}.csv")
    df_clean.to_csv(cleaned_csv, index=False, encoding="utf-8")
    log.info(f"Cleaned CSV saved: {cleaned_csv}")
    print(f"  Cleaned CSV   : {cleaned_csv}")

    # -- HTML report -----------------------------------------------------------
    _header("GENERATING HTML REPORT")
    report_path = generate_html_report(
        stats_before  = stats_before,
        stats_after   = stats_after,
        validation    = validation,
        steps_log     = steps_log,
        charts_before = charts_before,
        charts_after  = charts_after,
        output_dir    = OUTPUT_DIR,
        source_file   = os.path.basename(file_path),
    )
    print(f"  HTML Report   : {report_path}")
    print(f"  Log File      : {log_file}")

    _header("PIPELINE COMPLETE")
    print(f"""
  All outputs saved to: {OUTPUT_DIR}
    - cleaned_data_{ts}.csv
    - cleaning_report_{ts}.html
    - pipeline_log_{ts}.txt

  Open the HTML report in your browser for the full visual summary.
  """)


# ==============================================================================
if __name__ == "__main__":
    run_pipeline()
