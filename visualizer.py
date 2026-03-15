"""
visualizer.py
-------------
Generates all before/after visualizations for the pipeline.
Charts are returned as base64 strings for embedding in the HTML report.
Interactive display is handled separately by pipeline.py (_display_charts).
"""

import io
import base64
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Silent backend for base64 only — display handled by pipeline.py
import matplotlib.pyplot as plt
import seaborn as sns
import logging

warnings.filterwarnings("ignore")
logger = logging.getLogger("DataCleaningPipeline")

# ── Shared style ──────────────────────────────────────────────────────────────
BG_COLOR   = "#0f1117"
TEXT_COLOR = "#e0e0e0"
ACCENT     = "#4fc3f7"
GRID_COLOR = "#2a2a3d"

def _style():
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor":   BG_COLOR,
        "axes.edgecolor":   GRID_COLOR,
        "axes.labelcolor":  TEXT_COLOR,
        "xtick.color":      TEXT_COLOR,
        "ytick.color":      TEXT_COLOR,
        "text.color":       TEXT_COLOR,
        "grid.color":       GRID_COLOR,
        "axes.titlecolor":  TEXT_COLOR,
        "axes.titlesize":   11,
        "axes.labelsize":   9,
        "font.family":      "DejaVu Sans",
    })

def _to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=BG_COLOR)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


# ==============================================================================
# MISSING VALUES HEATMAP
# ==============================================================================

def plot_missing_heatmap_display(df: pd.DataFrame, title_prefix: str = "Before") -> str:
    """
    Renders missing values heatmap.

    FIX: Instead of relying on seaborn heatmap boolean colormap (which can render
    all-same-color due to normalization collapse), we build a custom RGBA image
    directly from the boolean mask. Red pixel = missing, dark pixel = present.
    This guarantees missing cells are always visible regardless of dataset size.
    """
    _style()
    missing_mask = df.isnull()
    has_missing  = missing_mask.any().any()

    fig, ax = plt.subplots(figsize=(max(10, len(df.columns) * 0.9), 5))

    if has_missing:
        # Build RGBA image manually — guarantees red = missing, dark = present
        mask_arr = missing_mask.values.astype(float)  # shape: (rows, cols)

        # Sample rows if dataset is very large (keep visual clear)
        if mask_arr.shape[0] > 500:
            step     = max(1, mask_arr.shape[0] // 500)
            mask_arr = mask_arr[::step]

        # RGBA: missing = bright red (#ef5350), present = dark (#1e1e2e)
        rgba = np.zeros((*mask_arr.shape, 4))
        rgba[mask_arr == 0] = [0.118, 0.118, 0.180, 1.0]   # dark present
        rgba[mask_arr == 1] = [0.937, 0.325, 0.314, 1.0]   # red missing

        ax.imshow(rgba, aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(df.columns)))
        ax.set_xticklabels(df.columns, rotation=45, ha="right", fontsize=9,
                           color=TEXT_COLOR)
        ax.set_yticks([])

        # Annotate each column with its missing count
        for j, col in enumerate(df.columns):
            cnt = missing_mask[col].sum()
            if cnt > 0:
                ax.text(j, -0.5, f"{cnt}", ha="center", va="bottom",
                        fontsize=8, color="#ef5350", fontweight="bold")

        ax.set_title(
            f"[{title_prefix}] Missing Values Heatmap  "
            f"(Red = Missing  |  Total missing cells: {missing_mask.values.sum():,})",
            pad=14, color=TEXT_COLOR
        )

        # Add a simple legend
        from matplotlib.patches import Patch
        legend = [
            Patch(facecolor="#ef5350", label="Missing"),
            Patch(facecolor="#1e1e2e", label="Present"),
        ]
        ax.legend(handles=legend, loc="upper right", framealpha=0.3,
                  facecolor=BG_COLOR, edgecolor=GRID_COLOR)
    else:
        ax.set_facecolor(BG_COLOR)
        ax.text(0.5, 0.5, "No Missing Values Found  ✓",
                ha="center", va="center", fontsize=16,
                color="#66bb6a", fontweight="bold", transform=ax.transAxes)
        ax.set_title(f"[{title_prefix}] Missing Values Heatmap", pad=14)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    return _to_b64(fig)


# ==============================================================================
# NUMERIC DISTRIBUTIONS
# ==============================================================================

def plot_numeric_distributions(df: pd.DataFrame, title_prefix: str = "Before") -> str:
    _style()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return ""

    n            = len(num_cols)
    cols_per_row = 4
    rows         = (n + cols_per_row - 1) // cols_per_row
    fig, axes    = plt.subplots(rows, cols_per_row,
                                figsize=(cols_per_row * 4, rows * 3.2))
    axes = np.array(axes).flatten()

    for i, col in enumerate(num_cols):
        data = df[col].dropna()
        axes[i].hist(data, bins=30, color=ACCENT, alpha=0.8, edgecolor="#1a1a2e")
        axes[i].set_title(col)
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Count")
        axes[i].grid(True, alpha=0.3)
        # Annotate with missing count
        missing = df[col].isnull().sum()
        if missing > 0:
            axes[i].set_xlabel(f"Value  (missing: {missing:,})", color="#ef5350")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"[{title_prefix}] Numeric Distributions", fontsize=13,
                 color=TEXT_COLOR, y=1.01)
    plt.tight_layout()
    return _to_b64(fig)


# ==============================================================================
# BOXPLOTS
# ==============================================================================

def plot_boxplots(df: pd.DataFrame, title_prefix: str = "Before") -> str:
    _style()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return ""

    n            = len(num_cols)
    cols_per_row = 4
    rows         = (n + cols_per_row - 1) // cols_per_row
    fig, axes    = plt.subplots(rows, cols_per_row,
                                figsize=(cols_per_row * 4, rows * 3.2))
    axes = np.array(axes).flatten()

    for i, col in enumerate(num_cols):
        data = df[col].dropna()
        axes[i].boxplot(
            data, patch_artist=True,
            medianprops=dict(color="#ef5350", linewidth=2),
            boxprops=dict(facecolor="#1a237e", color=ACCENT),
            whiskerprops=dict(color=ACCENT),
            capprops=dict(color=ACCENT),
            flierprops=dict(markerfacecolor="#ef5350", marker="o",
                            markersize=4, alpha=0.6)
        )
        axes[i].set_title(col)
        axes[i].set_ylabel("Value")
        axes[i].grid(True, alpha=0.3, axis="y")

        # Annotate outlier count
        Q1  = data.quantile(0.25)
        Q3  = data.quantile(0.75)
        IQR = Q3 - Q1
        out = ((data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)).sum()
        if out > 0:
            axes[i].set_xlabel(f"~{out:,} outlier(s)", color="#ef5350")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"[{title_prefix}] Boxplots — Dots = Outliers", fontsize=13,
                 color=TEXT_COLOR, y=1.01)
    plt.tight_layout()
    return _to_b64(fig)


# ==============================================================================
# CATEGORICAL COUNTS
# ==============================================================================

def plot_categorical_counts(df: pd.DataFrame, title_prefix: str = "Before",
                            max_cats: int = 10) -> str:
    _style()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not cat_cols:
        return ""

    n            = len(cat_cols)
    cols_per_row = 3
    rows         = (n + cols_per_row - 1) // cols_per_row
    fig, axes    = plt.subplots(rows, cols_per_row,
                                figsize=(cols_per_row * 5, rows * 3.5))
    axes = np.array(axes).flatten()

    colors = plt.cm.get_cmap("cool")(np.linspace(0.3, 0.9, max_cats))

    for i, col in enumerate(cat_cols):
        counts = df[col].value_counts().head(max_cats)
        axes[i].bar(range(len(counts)), counts.values,
                    color=colors[:len(counts)], edgecolor="#1a1a2e", alpha=0.85)
        axes[i].set_xticks(range(len(counts)))
        axes[i].set_xticklabels(counts.index, rotation=35, ha="right", fontsize=8)
        axes[i].set_title(f"{col}  (top {max_cats})")
        axes[i].set_ylabel("Count")
        axes[i].grid(True, alpha=0.3, axis="y")
        # Show missing count if any
        missing = df[col].isnull().sum()
        if missing > 0:
            axes[i].set_xlabel(f"missing: {missing:,}", color="#ef5350")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"[{title_prefix}] Categorical Value Counts", fontsize=13,
                 color=TEXT_COLOR, y=1.01)
    plt.tight_layout()
    return _to_b64(fig)


# ==============================================================================
# COMPARISON CHARTS (before vs after)
# ==============================================================================

def plot_duplicate_summary(dup_before: int, dup_after: int) -> str:
    _style()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Before", "After"], [dup_before, dup_after],
                  color=["#ef5350", "#66bb6a"], edgecolor="#1a1a2e", width=0.4)
    ax.set_title("Duplicate Rows — Before vs After", pad=12)
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, [dup_before, dup_after]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(dup_before, dup_after, 1) * 0.02,
                f"{val:,}", ha="center", va="bottom", fontsize=12, color=TEXT_COLOR)
    plt.tight_layout()
    return _to_b64(fig)


def plot_missing_summary(missing_before: int, missing_after: int) -> str:
    _style()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Before", "After"], [missing_before, missing_after],
                  color=["#ef5350", "#66bb6a"], edgecolor="#1a1a2e", width=0.4)
    ax.set_title("Missing Values — Before vs After", pad=12)
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, [missing_before, missing_after]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(missing_before, missing_after, 1) * 0.02,
                f"{val:,}", ha="center", va="bottom", fontsize=12, color=TEXT_COLOR)
    plt.tight_layout()
    return _to_b64(fig)


def plot_row_col_comparison(before_shape: tuple, after_shape: tuple) -> str:
    _style()
    categories  = ["Rows", "Columns"]
    before_vals = [before_shape[0], before_shape[1]]
    after_vals  = [after_shape[0],  after_shape[1]]

    x     = np.arange(len(categories))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    b1 = ax.bar(x - width/2, before_vals, width, label="Before",
                color="#ef5350", alpha=0.85)
    b2 = ax.bar(x + width/2, after_vals,  width, label="After",
                color="#66bb6a", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_title("Shape Comparison — Before vs After", pad=12)
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(before_vals) * 0.01,
                f"{int(bar.get_height()):,}",
                ha="center", va="bottom", fontsize=9, color=TEXT_COLOR)
    plt.tight_layout()
    return _to_b64(fig)


# ==============================================================================
# MAIN ENTRY POINTS
# ==============================================================================

def generate_before_charts(df: pd.DataFrame) -> dict:
    logger.info("  -> Generating before-cleaning visualizations...")
    charts = {
        "missing_heatmap_before":       plot_missing_heatmap_display(df, "Before"),
        "numeric_distributions_before": plot_numeric_distributions(df, "Before"),
        "boxplots_before":              plot_boxplots(df, "Before"),
        "categorical_counts_before":    plot_categorical_counts(df, "Before"),
    }
    logger.info("  -> Before charts ready.")
    return charts


def generate_after_charts(df_before: pd.DataFrame, df_after: pd.DataFrame,
                          dup_before: int, dup_after: int,
                          missing_before: int, missing_after: int) -> dict:
    logger.info("  -> Generating after-cleaning visualizations...")
    charts = {
        "missing_heatmap_after":       plot_missing_heatmap_display(df_after, "After"),
        "numeric_distributions_after": plot_numeric_distributions(df_after, "After"),
        "boxplots_after":              plot_boxplots(df_after, "After"),
        "categorical_counts_after":    plot_categorical_counts(df_after, "After"),
        "duplicate_comparison":        plot_duplicate_summary(dup_before, dup_after),
        "missing_comparison":          plot_missing_summary(missing_before, missing_after),
        "shape_comparison":            plot_row_col_comparison(df_before.shape, df_after.shape),
    }
    logger.info("  -> After charts ready.")
    return charts
