"""
reporter.py
-----------
Generates a professional HTML report with:
- Before and after stats comparison
- All charts embedded as base64 images
- Validation results
- Step-by-step log summary
Saved to the output folder.
"""

import os
from datetime import datetime
import logging

logger = logging.getLogger("DataCleaningPipeline")


def _chart_section(title: str, b64: str, description: str = "") -> str:
    """Returns an HTML block for a single chart with title and description."""
    if not b64:
        return ""
    desc_html = f'<p class="chart-desc">{description}</p>' if description else ""
    return f"""
    <div class="chart-card">
        <h3>{title}</h3>
        {desc_html}
        <img src="data:image/png;base64,{b64}" alt="{title}" />
    </div>
    """


def generate_html_report(
    stats_before: dict,
    stats_after: dict,
    validation: dict,
    steps_log: list,
    charts_before: dict,
    charts_after: dict,
    output_dir: str,
    source_file: str,
) -> str:
    """
    Builds a full HTML report and saves it to output_dir.
    Returns the path to the saved report.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_filename = os.path.join(
        output_dir,
        f"cleaning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    )

    # ── Stats table rows ──────────────────────────────────────────────────────
    def stat_row(label, before_val, after_val, good_direction="lower"):
        try:
            b = float(str(before_val).replace(",", ""))
            a = float(str(after_val).replace(",", ""))
            if good_direction == "lower":
                tag = "improved" if a < b else ("same" if a == b else "worse")
            else:
                tag = "improved" if a > b else ("same" if a == b else "worse")
        except Exception:
            tag = "same"

        icons = {"improved": "✅", "same": "➖", "worse": "⚠️"}
        return f"""
        <tr class="{tag}">
            <td>{label}</td>
            <td>{before_val}</td>
            <td>{after_val}</td>
            <td class="status-icon">{icons[tag]}</td>
        </tr>"""

    rows_diff     = stats_before["rows"] - stats_after["rows"]
    cols_diff     = stats_before["cols"] - stats_after["cols"]
    missing_diff  = stats_before["missing"] - stats_after["missing"]
    dup_diff      = stats_before["duplicates"] - stats_after["duplicates"]

    stats_table = f"""
    <table class="stats-table">
        <thead>
            <tr><th>Metric</th><th>Before</th><th>After</th><th>Status</th></tr>
        </thead>
        <tbody>
            {stat_row("Total Rows",         stats_before['rows'],       stats_after['rows'],       "lower")}
            {stat_row("Total Columns",      stats_before['cols'],       stats_after['cols'],       "lower")}
            {stat_row("Missing Values",     stats_before['missing'],    stats_after['missing'],    "lower")}
            {stat_row("Duplicate Rows",     stats_before['duplicates'], stats_after['duplicates'], "lower")}
            {stat_row("Memory Usage (KB)",  stats_before['memory_kb'],  stats_after['memory_kb'],  "lower")}
        </tbody>
    </table>
    """

    # ── Validation block ──────────────────────────────────────────────────────
    val_class  = "passed" if validation["passed"] else "failed"
    val_icon   = "✅ PASSED" if validation["passed"] else "⚠️ WARNINGS FOUND"
    val_issues = "".join(f"<li>{i}</li>" for i in validation.get("issues", []))
    val_block  = f"""
    <div class="validation-box {val_class}">
        <h3>Data Validation — {val_icon}</h3>
        <p>Final rows: {validation['total_rows']} &nbsp;|&nbsp;
           Final columns: {validation['total_columns']} &nbsp;|&nbsp;
           Nulls remaining: {validation['null_count']} &nbsp;|&nbsp;
           Duplicates remaining: {validation['duplicate_count']}</p>
        {"<ul>" + val_issues + "</ul>" if val_issues else "<p>All checks passed. Dataset is clean.</p>"}
    </div>
    """

    # ── Steps log ────────────────────────────────────────────────────────────
    steps_html = "".join(f"<li>{s}</li>" for s in steps_log)

    # ── Chart sections ────────────────────────────────────────────────────────
    before_charts_html = f"""
    <h2>📊 Before Cleaning — Visual Overview</h2>
    <p class="section-desc">
        These charts show the raw state of your dataset before any cleaning was applied.
        Use them to understand your data's original structure, distribution, and quality issues.
    </p>
    <div class="charts-grid">
        {_chart_section("Missing Values Heatmap (Before)",
            charts_before.get("missing_heatmap_before",""),
            "Red cells indicate missing data. The more red, the more nulls in that column.")}
        {_chart_section("Numeric Distributions (Before)",
            charts_before.get("numeric_distributions_before",""),
            "Histograms showing the frequency distribution of each numeric column before cleaning.")}
        {_chart_section("Boxplots — Outlier View (Before)",
            charts_before.get("boxplots_before",""),
            "Box plots reveal outliers (dots beyond the whiskers) in each numeric column.")}
        {_chart_section("Categorical Value Counts (Before)",
            charts_before.get("categorical_counts_before",""),
            "Bar charts showing the most frequent values in each categorical column.")}
    </div>
    """

    after_charts_html = f"""
    <h2>📊 After Cleaning — Visual Overview</h2>
    <p class="section-desc">
        These charts show your dataset after the full cleaning pipeline was applied.
        Compare with the before charts to see exactly what changed.
    </p>
    <div class="charts-grid">
        {_chart_section("Missing Values Heatmap (After)",
            charts_after.get("missing_heatmap_after",""),
            "After imputation, this map should show no red — all nulls filled.")}
        {_chart_section("Numeric Distributions (After)",
            charts_after.get("numeric_distributions_after",""),
            "Distributions after cleaning — outliers removed, types corrected.")}
        {_chart_section("Boxplots — Outlier View (After)",
            charts_after.get("boxplots_after",""),
            "Boxplots after IQR-based outlier removal. Fewer extreme dots expected.")}
        {_chart_section("Categorical Value Counts (After)",
            charts_after.get("categorical_counts_after",""),
            "Value counts after typo fixing — similar values should now be merged.")}
    </div>

    <h2>📈 Before vs After Comparison Charts</h2>
    <p class="section-desc">
        Direct comparison charts showing exactly how much the pipeline improved your dataset.
    </p>
    <div class="charts-grid three-col">
        {_chart_section("Duplicate Rows Comparison",
            charts_after.get("duplicate_comparison",""),
            "Red = before, Green = after. Goal is zero duplicates after.")}
        {_chart_section("Missing Values Comparison",
            charts_after.get("missing_comparison",""),
            "Total null count before and after imputation.")}
        {_chart_section("Shape Comparison",
            charts_after.get("shape_comparison",""),
            "How rows and columns changed after dropping constants, high-cardinality cols, and outliers.")}
    </div>
    """

    # ── Full HTML ─────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <title>Data Cleaning Pipeline Report</title>
    <style>
        :root {{
            --bg:       #0f1117;
            --card:     #1a1a2e;
            --border:   #2a2a3d;
            --accent:   #4fc3f7;
            --green:    #66bb6a;
            --red:      #ef5350;
            --yellow:   #ffa726;
            --text:     #e0e0e0;
            --subtext:  #9e9e9e;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            padding: 30px 40px;
            line-height: 1.7;
        }}
        header {{
            border-bottom: 2px solid var(--accent);
            padding-bottom: 20px;
            margin-bottom: 35px;
        }}
        header h1 {{
            font-size: 2rem;
            color: var(--accent);
            letter-spacing: 1px;
        }}
        header .meta {{
            color: var(--subtext);
            font-size: 0.9rem;
            margin-top: 6px;
        }}
        h2 {{
            font-size: 1.35rem;
            color: var(--accent);
            margin: 40px 0 12px;
            border-left: 4px solid var(--accent);
            padding-left: 12px;
        }}
        h3 {{
            font-size: 1rem;
            color: var(--text);
            margin-bottom: 8px;
        }}
        .section-desc {{
            color: var(--subtext);
            font-size: 0.92rem;
            margin-bottom: 20px;
        }}
        /* ── Stats Table ── */
        .stats-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0 30px;
            font-size: 0.95rem;
        }}
        .stats-table thead tr {{
            background: var(--border);
            color: var(--accent);
        }}
        .stats-table th, .stats-table td {{
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        .stats-table tr.improved {{ background: rgba(102,187,106,0.07); }}
        .stats-table tr.worse    {{ background: rgba(239,83,80,0.07); }}
        .status-icon {{ font-size: 1.1rem; text-align: center; }}
        /* ── Charts ── */
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 24px;
            margin-bottom: 30px;
        }}
        .charts-grid.three-col {{
            grid-template-columns: repeat(3, 1fr);
        }}
        .chart-card {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 18px;
        }}
        .chart-card img {{
            width: 100%;
            border-radius: 6px;
            margin-top: 8px;
        }}
        .chart-desc {{
            color: var(--subtext);
            font-size: 0.83rem;
            margin-bottom: 8px;
        }}
        /* ── Validation ── */
        .validation-box {{
            border-radius: 10px;
            padding: 20px 24px;
            margin: 20px 0 30px;
            border: 1px solid;
        }}
        .validation-box.passed {{
            background: rgba(102,187,106,0.1);
            border-color: var(--green);
        }}
        .validation-box.failed {{
            background: rgba(239,83,80,0.1);
            border-color: var(--red);
        }}
        .validation-box h3 {{ margin-bottom: 10px; }}
        .validation-box ul {{ margin: 10px 0 0 20px; }}
        /* ── Steps log ── */
        .steps-log {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 20px 24px;
            margin: 12px 0 30px;
        }}
        .steps-log ol {{
            padding-left: 22px;
            font-size: 0.9rem;
            color: var(--subtext);
            line-height: 2;
        }}
        .steps-log li {{ border-bottom: 1px solid var(--border); padding: 4px 0; }}
        /* ── Footer ── */
        footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid var(--border);
            color: var(--subtext);
            font-size: 0.85rem;
            text-align: center;
        }}
    </style>
</head>
<body>
    <header>
        <h1>🧹 Data Cleaning Pipeline Report</h1>
        <div class="meta">
            Generated: {timestamp} &nbsp;|&nbsp;
            Source file: <code>{source_file}</code>
        </div>
    </header>

    <h2>📋 Before vs After Summary</h2>
    <p class="section-desc">
        A complete comparison of your dataset's key metrics before and after the cleaning pipeline.
        Green rows indicate improvement. Blue rows indicate no change.
    </p>
    {stats_table}

    {val_block}

    {before_charts_html}

    {after_charts_html}

    <h2>🔄 Pipeline Steps Executed</h2>
    <p class="section-desc">
        Every step the pipeline executed, in order. This log gives you full transparency
        into what was changed and why.
    </p>
    <div class="steps-log">
        <ol>
            {steps_html}
        </ol>
    </div>

    <footer>
        Data Cleaning Pipeline &mdash; Built with Python &amp; Matplotlib &mdash;
        All charts generated from your actual data &mdash; {timestamp}
    </footer>
</body>
</html>"""

    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"  → HTML report saved: {report_filename}")
    return report_filename
