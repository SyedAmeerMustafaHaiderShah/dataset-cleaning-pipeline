"""
cleaner.py
----------
Contains all individual data cleaning methods.
Internal logic is preserved exactly as designed.
Each method logs what it does and how many rows were affected.
"""

import re
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("DataCleaningPipeline")


# ─────────────────────────────────────────────
# COLUMN VERIFICATION
# ─────────────────────────────────────────────

def verify_columns(df: pd.DataFrame, user_list: list) -> list:
    """
    Checks which user-provided column names actually exist in the DataFrame.
    Returns only the valid ones and warns about the invalid ones.
    """
    dataset_cols = set(df.columns)
    valid = []
    invalid = []

    for col in user_list:
        if col in dataset_cols:
            valid.append(col)
        else:
            invalid.append(col)

    if invalid:
        logger.warning(
            f"⚠️  WARNING — The following column(s) were NOT found in the dataset. "
            f"Please check for typos: {invalid}"
        )
    if valid:
        logger.info(f"✅ Verified columns: {valid}")

    return valid


# ─────────────────────────────────────────────
# STEP 1 — LOWERCASE COLUMN NAMES
# ─────────────────────────────────────────────

def lowercase_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercases all column names."""
    before = list(df.columns)
    df.columns = [col.lower() for col in df.columns]
    after = list(df.columns)
    changed = [(b, a) for b, a in zip(before, after) if b != a]
    logger.info(f"  → Lowercased {len(changed)} column name(s): {changed if changed else 'none needed'}")
    return df


# ─────────────────────────────────────────────
# STEP 2 — CLEAN WHITESPACE
# ─────────────────────────────────────────────

def clean_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Strips leading/trailing spaces and collapses internal multiple spaces."""
    str_cols = df.select_dtypes(include=["object"]).columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].str.replace(r"\s+", " ", regex=True)
    logger.info(f"  → Whitespace cleaned in {len(str_cols)} string column(s).")
    return df


# ─────────────────────────────────────────────
# STEP 3 — FIX DATE INCONSISTENCY
# ─────────────────────────────────────────────

def fix_date_inconsistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects columns with date-related names and standardizes
    them into a unified datetime format.
    Works with both object and pandas StringDtype columns.
    """
    date_keywords = ["date", "time", "timestamp", "year", "dob"]
    fixed = []

    for col in df.columns:
        # Check both classic object dtype and newer pandas StringDtype
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == "object":
            if any(key in col.lower() for key in date_keywords):
                df[col] = pd.to_datetime(df[col], errors="coerce")
                fixed.append(col)

    if fixed:
        logger.info(f"  → Standardized date format in column(s): {fixed}")
    else:
        logger.info("  → No date columns detected.")
    return df


# ─────────────────────────────────────────────
# STEP 4 — DROP CONSTANT COLUMNS
# ─────────────────────────────────────────────

def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drops columns that contain only one unique value (zero variance)."""
    before_cols = df.shape[1]
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    df = df.drop(columns=constant_cols)
    logger.info(
        f"  → Dropped {len(constant_cols)} constant column(s): "
        f"{constant_cols if constant_cols else 'none found'}"
    )
    return df


# ─────────────────────────────────────────────
# STEP 5 — DROP HIGH CARDINALITY COLUMNS
# ─────────────────────────────────────────────

def drop_high_cardinality(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """
    Drops columns where the ratio of unique values to total rows
    is above the threshold (e.g., ID columns, random strings).
    """
    high_card_cols = [
        col for col in df.columns
        if (df[col].nunique() / len(df)) >= threshold
    ]
    df = df.drop(columns=high_card_cols)
    logger.info(
        f"  → Dropped {len(high_card_cols)} high-cardinality column(s) "
        f"(>={threshold*100:.0f}% unique): "
        f"{high_card_cols if high_card_cols else 'none found'}"
    )
    return df


# ─────────────────────────────────────────────
# STEP 6 — REMOVE ALL DUPLICATES
# ─────────────────────────────────────────────

def remove_all_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Removes fully duplicate rows across all columns."""
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    logger.info(f"  → Removed {removed} fully duplicate row(s). Rows remaining: {len(df)}")
    return df


# ─────────────────────────────────────────────
# STEP 7 — SELECTIVE DUPLICATE REMOVAL (by user columns)
# ─────────────────────────────────────────────

def remove_duplicates_by_list(df: pd.DataFrame, confirmed_list: list) -> pd.DataFrame:
    """
    Removes duplicates based on specific user-selected columns sequentially.
    Only runs if the user provided valid column names.
    """
    if not confirmed_list:
        logger.info("  → No columns selected for selective duplicate removal. Skipping.")
        return df

    total_removed = 0
    for col in confirmed_list:
        before = len(df)
        df = df.drop_duplicates(subset=[col])
        removed = before - len(df)
        total_removed += removed
        logger.info(f"     Column '{col}': removed {removed} selective duplicate(s).")

    logger.info(f"  → Total rows removed in selective duplicate step: {total_removed}")
    return df


# ─────────────────────────────────────────────
# STEP 8 — STRIP SYMBOLS CONDITIONALLY
# ─────────────────────────────────────────────

def strip_symbols_conditionally(df: pd.DataFrame, threshold: float = 0.70) -> pd.DataFrame:
    """
    For each column, if >= threshold% of values look like numbers
    after stripping symbols, apply the symbol stripping.
    """
    cleaned_cols = []
    for col in df.columns:
        # Skip columns that are already proper numeric — no symbol stripping needed
        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        cleaned_col = df[col].apply(
            lambda x: re.sub(r"[^0-9.\-]", "", str(x)) if pd.notna(x) else x
        )
        valid_numeric_count = pd.to_numeric(cleaned_col, errors="coerce").notna().sum()
        if (valid_numeric_count / len(df)) >= threshold:
            # Convert immediately to numeric so downstream steps see correct dtype
            df[col] = pd.to_numeric(cleaned_col, errors="coerce")
            cleaned_cols.append(col)

    logger.info(
        f"  → Symbol stripping applied to {len(cleaned_cols)} column(s): "
        f"{cleaned_cols if cleaned_cols else 'none qualified'}"
    )
    return df


# ─────────────────────────────────────────────
# STEP 9 — FIX MAJORITY TYPE
# ─────────────────────────────────────────────

def fix_majority_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each column, detects the majority data type and enforces it.
    Numeric majority → pd.to_numeric. Otherwise → string.
    """
    for col in df.columns:
        type_counts = df[col].apply(lambda x: type(x).__name__).value_counts()
        majority_type = type_counts.idxmax()

        if majority_type in ["int", "float", "int64", "float64"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = df[col].astype(str)

    logger.info(f"  → Majority type enforcement applied to all {len(df.columns)} column(s).")
    return df


# ─────────────────────────────────────────────
# STEP 10 — SMART IMPUTE
# ─────────────────────────────────────────────

def smart_impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing values intelligently:
    - Numeric columns: median (if skewed > 1) or mean
    - Categorical columns: mode (or 'Unknown' if no mode)
    """
    num_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    cat_cols = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]

    total_filled = 0

    for col in num_cols:
        if df[col].isnull().any():
            nulls = df[col].isnull().sum()
            # Compute skew BEFORE filling so log message matches what was actually used
            is_skewed = abs(df[col].skew()) > 1
            fill_val = df[col].median() if is_skewed else df[col].mean()
            df[col] = df[col].fillna(fill_val)
            total_filled += nulls
            logger.info(f"     Numeric '{col}': filled {nulls} null(s) with {'median' if is_skewed else 'mean'} = {fill_val:.4f}")

    for col in cat_cols:
        if df[col].isnull().any():
            nulls = df[col].isnull().sum()
            mode_series = df[col].mode()
            fill_val = mode_series[0] if not mode_series.empty else "Unknown"
            df[col] = df[col].fillna(fill_val)
            total_filled += nulls
            logger.info(f"     Categorical '{col}': filled {nulls} null(s) with mode = '{fill_val}'")

    logger.info(f"  → Smart imputation complete. Total values filled: {total_filled}")
    return df


# ─────────────────────────────────────────────
# STEP 11 — FIX TYPOS WITH RAPIDFUZZ
# ─────────────────────────────────────────────

def fix_typos_with_rapidfuzz(df: pd.DataFrame, threshold: int = 90) -> pd.DataFrame:
    """
    Detects and fixes typos in categorical string columns by matching
    rare values to high-frequency correct values using RapidFuzz similarity.
    """
    try:
        from rapidfuzz import process, utils, fuzz
    except ImportError:
        logger.warning("  ⚠️  rapidfuzz not installed. Skipping typo fixing step. Run: pip install rapidfuzz")
        return df

    text_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    total_fixed = 0

    for col in text_cols:
        counts = df[col].value_counts()
        unique_values = counts.index.tolist()
        replacement_map = {}

        for i in range(len(unique_values) - 1, 0, -1):
            target = unique_values[i]
            choices = unique_values[:i]
            match = process.extractOne(
                target,
                choices,
                scorer=fuzz.WRatio,
                processor=utils.default_process,
                score_cutoff=threshold,
            )
            if match:
                replacement_map[target] = match[0]

        if replacement_map:
            df[col] = df[col].replace(replacement_map)
            total_fixed += len(replacement_map)
            logger.info(f"     '{col}': fixed {len(replacement_map)} typo(s) → {replacement_map}")

    logger.info(f"  → Typo fixing complete. Total replacements: {total_fixed}")
    return df


# ─────────────────────────────────────────────
# STEP 12 — REMOVE OUTLIERS (IQR)
# ─────────────────────────────────────────────

def remove_outliers(df: pd.DataFrame, col_list: list) -> pd.DataFrame:
    """
    Removes rows with outliers in the specified numeric columns using IQR method.
    Only runs if the user provided valid column names.
    """
    if not col_list:
        logger.info("  → No columns selected for outlier removal. Skipping.")
        return df

    before = len(df)
    for col in col_list:
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.warning(f"  ⚠️  Column '{col}' is not numeric. Skipping outlier removal for it.")
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        rows_before = len(df)
        df = df[(df[col] >= lower) & (df[col] <= upper)]
        removed = rows_before - len(df)
        logger.info(f"     Column '{col}': removed {removed} outlier row(s). Bounds: [{lower:.4f}, {upper:.4f}]")

    total_removed = before - len(df)
    logger.info(f"  → Total rows removed in outlier step: {total_removed}")
    return df


# ─────────────────────────────────────────────
# DATA VALIDATION
# ─────────────────────────────────────────────

def validate_cleaned_data(df: pd.DataFrame) -> dict:
    """
    Self-verifying step. Checks that the cleaned DataFrame
    truly has no nulls, no full duplicates, etc.
    Returns a validation report dictionary.
    """
    results = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "null_count": int(df.isnull().sum().sum()),
        "duplicate_count": int(df.duplicated().sum()),
        "passed": True,
        "issues": [],
    }

    if results["null_count"] > 0:
        results["passed"] = False
        results["issues"].append(f"{results['null_count']} null value(s) still present.")

    if results["duplicate_count"] > 0:
        results["passed"] = False
        results["issues"].append(f"{results['duplicate_count']} duplicate row(s) still present.")

    if results["passed"]:
        logger.info("  ✅ Validation PASSED — Dataset is clean.")
    else:
        for issue in results["issues"]:
            logger.warning(f"  ⚠️  Validation WARNING: {issue}")

    return results
