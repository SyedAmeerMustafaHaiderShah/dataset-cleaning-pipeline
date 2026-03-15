"""
app.py
------
FastAPI wrapper around the Data Cleaning Pipeline.
Fully integrated with visualizer.py and reporter.py.

Run with:
    python -m uvicorn app:app --reload

Then open:
    http://127.0.0.1:8000/docs
"""

import uuid
import json
import shutil
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from cleaner import (
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
    verify_columns,
)
from visualizer import generate_before_charts, generate_after_charts
from reporter import generate_html_report

app = FastAPI(
    title="Data Cleaning Pipeline API",
    description="Upload a CSV or Excel file, configure cleaning options, run the pipeline, and download the cleaned dataset.",
    version="1.0.0",
)

TEMP_DIR = Path("temp")
TEMP_DIR.mkdir(exist_ok=True)

active_session = {"session_id": None}


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_session_path(session_id: str) -> Path:
    path = TEMP_DIR / session_id
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. Please upload a file first."
        )
    return path


def load_df(session_id: str) -> pd.DataFrame:
    session_path = get_session_path(session_id)
    csv_path = session_path / "raw.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Raw file not found in session.")
    return pd.read_csv(csv_path)


def save_config(session_id: str, key: str, value):
    session_path = get_session_path(session_id)
    config_path = session_path / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {}
    config[key] = value
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def load_config(session_id: str) -> dict:
    session_path = get_session_path(session_id)
    config_path = session_path / "config.json"
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return json.load(f)


def cleanup_session(session_id: str):
    session_path = TEMP_DIR / session_id
    shutil.rmtree(session_path, ignore_errors=True)


def _df_stats(df: pd.DataFrame) -> dict:
    """
    Builds the stats dict in the exact format reporter.py expects.
    Keys: rows, cols, missing, duplicates, memory_kb
    """
    return {
        "rows":       len(df),
        "cols":       len(df.columns),
        "missing":    int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "memory_kb":  round(df.memory_usage(deep=True).sum() / 1024, 2),
    }


# ── Pydantic models ───────────────────────────────────────────────────────────

class CardinalityConfig(BaseModel):
    session_id: str
    columns_to_drop: list[str]

class DuplicatesConfig(BaseModel):
    session_id: str
    columns: list[str]

class OutliersConfig(BaseModel):
    session_id: str
    columns: list[str]

class CorrectionsConfig(BaseModel):
    session_id: str
    corrections: dict


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 1 — Upload file
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Receives a CSV or Excel file.
    Deletes previous session automatically — saves server space.
    Returns session_id and dataset overview.
    """

    # Delete previous session if exists
    if active_session["session_id"]:
        cleanup_session(active_session["session_id"])
        active_session["session_id"] = None

    # Validate file type
    allowed_extensions = [".csv", ".xlsx", ".xls"]
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file_ext}'. Allowed: {allowed_extensions}"
        )

    # Create session folder
    session_id = str(uuid.uuid4())[:8]
    session_path = TEMP_DIR / session_id
    session_path.mkdir(parents=True, exist_ok=True)

    # Save raw file
    raw_path = session_path / f"raw{file_ext}"
    try:
        contents = await file.read()
        with open(raw_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        shutil.rmtree(session_path, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Read as DataFrame and save internal CSV
    try:
        df = pd.read_csv(raw_path) if file_ext == ".csv" else pd.read_excel(raw_path)
        df.to_csv(session_path / "raw.csv", index=False)
    except Exception as e:
        shutil.rmtree(session_path, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Could not read file: {str(e)}")

    active_session["session_id"] = session_id

    # Cardinality table
    cardinality_info = {}
    for col in df.columns:
        unique_count = df[col].nunique()
        unique_ratio = round(unique_count / len(df), 3)
        cardinality_info[col] = {
            "unique_values":    unique_count,
            "unique_ratio":     unique_ratio,
            "high_cardinality": unique_ratio >= 0.95
        }

    return {
        "success":    True,
        "session_id": session_id,
        "message":    f"File '{file.filename}' uploaded. Previous session cleared.",
        "dataset_info": {
            "rows":           len(df),
            "columns":        len(df.columns),
            "column_names":   list(df.columns),
            "missing_values": int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "memory_mb":      round(df.memory_usage(deep=True).sum() / 1024 / 1024, 3),
        },
        "cardinality_table": cardinality_info,
        "next_step": "Call GET /preview/{session_id} to see stats and before-cleaning charts."
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 2 — Preview (stats + before charts)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/preview/{session_id}")
def preview(session_id: str):
    """
    Returns before-cleaning statistics AND generates before-cleaning charts.
    Charts are returned as base64 strings — embed directly in <img src="data:image/png;base64,...">
    No cleaning happens here.
    """
    df = load_df(session_id)

    # Per-column stats
    column_stats = {}
    for col in df.columns:
        col_stats = {
            "dtype":           str(df[col].dtype),
            "missing_count":   int(df[col].isnull().sum()),
            "missing_percent": round(df[col].isnull().mean() * 100, 2),
            "unique_values":   int(df[col].nunique()),
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            col_stats.update({
                "min":    round(float(df[col].min()), 4),
                "max":    round(float(df[col].max()), 4),
                "mean":   round(float(df[col].mean()), 4),
                "median": round(float(df[col].median()), 4),
            })
        else:
            top_values = df[col].value_counts().head(5).to_dict()
            col_stats["top_values"] = {str(k): int(v) for k, v in top_values.items()}
        column_stats[col] = col_stats

    # Generate before charts and save to session
    charts_before = {}
    chart_error = None
    try:
        charts_before = generate_before_charts(df)
        session_path = get_session_path(session_id)
        with open(session_path / "charts_before.json", "w") as f:
            json.dump(charts_before, f)
    except Exception as e:
        chart_error = str(e)

    return {
        "success":      True,
        "session_id":   session_id,
        "overview": {
            "rows":             len(df),
            "columns":          len(df.columns),
            "total_missing":    int(df.isnull().sum().sum()),
            "total_duplicates": int(df.duplicated().sum()),
        },
        "column_stats":  column_stats,
        "charts_before": charts_before,
        "chart_error":   chart_error,
        "next_step": "Configure cleaning options using POST /configure/* endpoints."
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 3 — Configure: Cardinality
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/configure/cardinality")
def configure_cardinality(config: CardinalityConfig):
    """Saves which high-cardinality columns to drop. Nothing dropped yet."""
    df = load_df(config.session_id)
    valid_cols = verify_columns(df, config.columns_to_drop)
    save_config(config.session_id, "cardinality_drop_cols", valid_cols)
    return {
        "success":   True,
        "session_id": config.session_id,
        "saved":     valid_cols,
        "message":   f"{len(valid_cols)} column(s) marked for cardinality drop.",
        "next_step": "Call POST /configure/duplicates"
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 4 — Configure: Duplicates
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/configure/duplicates")
def configure_duplicates(config: DuplicatesConfig):
    """Saves columns for selective duplicate removal. Nothing removed yet."""
    df = load_df(config.session_id)
    valid_cols = verify_columns(df, config.columns)
    save_config(config.session_id, "selective_dup_cols", valid_cols)
    return {
        "success":   True,
        "session_id": config.session_id,
        "saved":     valid_cols,
        "message":   f"{len(valid_cols)} column(s) saved for selective duplicate removal.",
        "next_step": "Call POST /configure/outliers"
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 5 — Configure: Outliers
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/configure/outliers")
def configure_outliers(config: OutliersConfig):
    """Saves columns for IQR outlier removal. Nothing removed yet."""
    df = load_df(config.session_id)
    valid_cols = verify_columns(df, config.columns)
    save_config(config.session_id, "outlier_cols", valid_cols)
    return {
        "success":   True,
        "session_id": config.session_id,
        "saved":     valid_cols,
        "message":   f"{len(valid_cols)} column(s) saved for outlier removal.",
        "next_step": "Call POST /configure/corrections or skip to POST /run/{session_id}"
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 6 — Configure: Manual Corrections
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/configure/corrections")
def configure_corrections(config: CorrectionsConfig):
    """
    Saves manual value corrections per column. Nothing corrected yet.
    Example:
    {
        "session_id": "a3f9b2c1",
        "corrections": {
            "city": {"N.Y.C": "New York", "Lo n d o n": "London"}
        }
    }
    """
    save_config(config.session_id, "manual_corrections", config.corrections)
    return {
        "success":   True,
        "session_id": config.session_id,
        "saved":     config.corrections,
        "message":   "Manual corrections saved.",
        "next_step": "Call POST /run/{session_id} to execute the full pipeline."
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 7 — Run pipeline
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/run/{session_id}")
def run_pipeline(session_id: str):
    """
    Executes all 12 cleaning steps.
    Generates before + after charts.
    Saves cleaned CSV and full HTML report to session folder.
    This is the ONLY endpoint where cleaning actually happens.
    """
    session_path = get_session_path(session_id)
    df_raw = load_df(session_id)
    config = load_config(session_id)

    cardinality_drop_cols = config.get("cardinality_drop_cols", [])
    selective_dup_cols    = config.get("selective_dup_cols", [])
    outlier_cols          = config.get("outlier_cols", [])
    manual_corrections    = config.get("manual_corrections", {})

    # Capture before stats in reporter.py format
    stats_before = _df_stats(df_raw)

    # Generate before charts (use cached if preview was called, else generate now)
    charts_before_path = session_path / "charts_before.json"
    if charts_before_path.exists():
        with open(charts_before_path, "r") as f:
            charts_before = json.load(f)
    else:
        charts_before = generate_before_charts(df_raw)

    df = df_raw.copy()
    steps_log = []

    try:
        df = lowercase_columns(df)
        steps_log.append("Column names lowercased.")

        df = clean_whitespace(df)
        steps_log.append("Whitespace stripped and collapsed in all string columns.")

        df = fix_date_inconsistency(df)
        steps_log.append("Date columns detected and standardized to datetime format.")

        df = drop_constant_columns(df)
        steps_log.append("Constant columns (zero variance) dropped automatically.")

        if cardinality_drop_cols:
            valid = [c for c in cardinality_drop_cols if c in df.columns]
            df = df.drop(columns=valid)
            steps_log.append(f"High-cardinality columns dropped: {valid}")
        else:
            steps_log.append("No high-cardinality columns selected for dropping.")

        before_dup = len(df)
        df = remove_all_duplicates(df)
        steps_log.append(f"Full duplicate rows removed: {before_dup - len(df)} row(s).")

        selective_dup_cols = [c for c in selective_dup_cols if c in df.columns]
        df = remove_duplicates_by_list(df, selective_dup_cols)
        steps_log.append(f"Selective duplicate removal on columns: {selective_dup_cols or 'none selected'}.")

        df = strip_symbols_conditionally(df)
        steps_log.append("Non-numeric symbols stripped from numeric-like columns.")

        df = fix_majority_type(df)
        steps_log.append("Majority data type enforced per column.")

        before_missing = int(df.isnull().sum().sum())
        df = smart_impute(df)
        after_missing = int(df.isnull().sum().sum())
        steps_log.append(f"Smart imputation: {before_missing - after_missing} value(s) filled. Numeric -> median/mean. Categorical -> mode.")

        if manual_corrections:
            for col, mapping in manual_corrections.items():
                if col in df.columns:
                    df[col] = df[col].replace(mapping)
            steps_log.append(f"Manual value corrections applied: {manual_corrections}")
        else:
            steps_log.append("No manual corrections configured.")

        df = fix_typos_with_rapidfuzz(df)
        steps_log.append("Typo fixing with RapidFuzz completed.")

        outlier_cols = [c for c in outlier_cols if c in df.columns]
        df = remove_outliers(df, outlier_cols)
        steps_log.append(f"Outlier removal (IQR) on columns: {outlier_cols or 'none selected'}.")

        df = remove_all_duplicates(df)
        steps_log.append("Final deduplication pass completed.")

        validation = validate_cleaned_data(df)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")

    # Save cleaned CSV
    cleaned_csv_path = session_path / "cleaned.csv"
    df.to_csv(cleaned_csv_path, index=False)

    # Capture after stats
    stats_after = _df_stats(df)

    # Generate after charts
    charts_after = generate_after_charts(
        df_before=df_raw,
        df_after=df,
        dup_before=stats_before["duplicates"],
        dup_after=stats_after["duplicates"],
        missing_before=stats_before["missing"],
        missing_after=stats_after["missing"],
    )

    # Save after charts to session
    with open(session_path / "charts_after.json", "w") as f:
        json.dump(charts_after, f)

    # Generate HTML report
    report_path = generate_html_report(
        stats_before=stats_before,
        stats_after=stats_after,
        validation=validation,
        steps_log=steps_log,
        charts_before=charts_before,
        charts_after=charts_after,
        output_dir=str(session_path),
        source_file=config.get("source_filename", "dataset.csv"),
    )

    # Rename report to fixed name so download endpoint can find it
    fixed_report_path = session_path / "report.html"
    Path(report_path).rename(fixed_report_path)

    return {
        "success":      True,
        "session_id":   session_id,
        "validation":   validation,
        "stats_before": stats_before,
        "stats_after":  stats_after,
        "rows_removed": stats_before["rows"] - stats_after["rows"],
        "next_step":    "Call GET /download/csv/{session_id} or GET /download/report/{session_id}"
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 8 — After-cleaning stats + charts
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/results/charts/{session_id}")
def results_charts(session_id: str):
    """
    Returns after-cleaning statistics and charts.
    Charts are base64 strings — embed in <img src="data:image/png;base64,...">
    Call after /run.
    """
    session_path = get_session_path(session_id)
    cleaned_path = session_path / "cleaned.csv"

    if not cleaned_path.exists():
        raise HTTPException(
            status_code=400,
            detail="Pipeline has not been run yet. Call POST /run/{session_id} first."
        )

    df = pd.read_csv(cleaned_path)

    # Load saved after charts if available
    charts_after_path = session_path / "charts_after.json"
    charts_after = {}
    if charts_after_path.exists():
        with open(charts_after_path, "r") as f:
            charts_after = json.load(f)

    column_stats = {}
    for col in df.columns:
        col_stats = {
            "dtype":         str(df[col].dtype),
            "missing_count": int(df[col].isnull().sum()),
            "unique_values": int(df[col].nunique()),
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            col_stats.update({
                "min":  round(float(df[col].min()), 4),
                "max":  round(float(df[col].max()), 4),
                "mean": round(float(df[col].mean()), 4),
            })
        column_stats[col] = col_stats

    return {
        "success":      True,
        "session_id":   session_id,
        "overview": {
            "rows":       len(df),
            "columns":    len(df.columns),
            "missing":    int(df.isnull().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
        },
        "column_stats": column_stats,
        "charts_after": charts_after,
        "next_step":    "Call GET /download/csv/{session_id} to download."
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 9 — Download cleaned CSV
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/download/csv/{session_id}")
def download_csv(session_id: str):
    """
    Returns the cleaned CSV file.
    Session stays alive until user uploads a new file.
    """
    session_path = get_session_path(session_id)
    cleaned_path = session_path / "cleaned.csv"

    if not cleaned_path.exists():
        raise HTTPException(
            status_code=400,
            detail="No cleaned file found. Run the pipeline first."
        )

    return FileResponse(
        path=str(cleaned_path),
        filename=f"cleaned_{session_id}.csv",
        media_type="text/csv"
    )


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 10 — Download HTML report
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/download/report/{session_id}")
def download_report(session_id: str):
    """
    Returns the full HTML cleaning report with embedded charts.
    Session stays alive until user uploads a new file.
    """
    session_path = get_session_path(session_id)
    report_path = session_path / "report.html"

    if not report_path.exists():
        raise HTTPException(
            status_code=400,
            detail="No report found. Run the pipeline first with POST /run/{session_id}."
        )

    return FileResponse(
        path=str(report_path),
        filename=f"report_{session_id}.html",
        media_type="text/html"
    )
