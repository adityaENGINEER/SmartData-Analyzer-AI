"""
Comprehensive Report and Export Generator for DataDoctor AI.
Generates structured Markdown/HTML diagnostic reports, CSV/Excel export bundles, and cleaning audit logs.
"""

import io
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime


class ReportGenerator:
    """Consolidates data profiling, cleaning operations, and analytics into shareable artifacts."""

    @staticmethod
    def generate_markdown_report(
        filename: str,
        profile_data: Dict[str, Any],
        quality_score: Dict[str, Any],
        cleaning_history: List[Dict[str, Any]],
        notes: Optional[str] = None,
    ) -> str:
        """Generates a structured, executive-ready Markdown data health report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        md = []
        md.append(f"# DataDoctor AI — Data Quality & Analytics Report")
        md.append(f"**Dataset:** `{filename}` | **Generated:** {timestamp}\n")
        md.append("---")

        # 1. Executive Summary & Quality Score
        score = quality_score.get("overall_score", 0.0)
        grade = quality_score.get("grade", "N/A")
        md.append(f"## 1. Executive Summary & Health Score")
        md.append(f"- **Overall Data Quality Score:** **{score} / 100** ({grade})")
        md.append(f"- **Total Rows:** {profile_data.get('total_rows', 0):,}")
        md.append(f"- **Total Columns:** {profile_data.get('total_cols', 0):,}")
        md.append(f"- **Missing Value Ratio:** {profile_data.get('missing_pct', 0.0)}% ({profile_data.get('missing_cells', 0):,} cells)")
        md.append(f"- **Duplicate Records:** {profile_data.get('duplicate_pct', 0.0)}% ({profile_data.get('duplicate_rows', 0):,} rows)\n")

        # 2. Quality Score Breakdown
        md.append("### Score Component Breakdown")
        comps = quality_score.get("components", {})
        md.append("| Dimension | Score | Max Points | Deduction Reason |")
        md.append("| :--- | :--- | :--- | :--- |")
        for name, comp in comps.items():
            ded = comp.get("deduction", 0.0)
            sc = comp.get("score", 0.0)
            mx = comp.get("max", 0.0)
            md.append(f"| {name.capitalize()} | {sc} | {mx} | -{ded} pts |")
        md.append("")

        # 3. Top Impact Issues
        top_impacts = quality_score.get("top_impacts", [])
        if top_impacts:
            md.append("### Primary Quality Bottlenecks")
            for imp in top_impacts:
                md.append(f"- {imp}")
            md.append("")

        # 4. Detected Issues Log
        issues = profile_data.get("issues", [])
        md.append(f"## 2. Detected Issues & Diagnostics ({len(issues)} total)")
        if not issues:
            md.append("No critical issues or warnings detected in this dataset.")
        else:
            md.append("| Issue Type | Column | Severity | Affected Count | Suggested Action |")
            md.append("| :--- | :--- | :--- | :--- | :--- |")
            for iss in issues:
                col_name = iss.get("column") or "Whole Dataset"
                md.append(f"| {iss.get('issue_type')} | `{col_name}` | {iss.get('severity')} | {iss.get('count')} ({iss.get('percentage')}%) | {iss.get('remediation')} |")
            md.append("")

        # 5. Cleaning Operations Applied
        md.append(f"## 3. Cleaning & Remediation Audit Trail")
        if not cleaning_history:
            md.append("No transformation or cleaning operations have been applied to this session yet.")
        else:
            md.append("| Step | Operation | Column | Rows Affected | Details |")
            md.append("| :--- | :--- | :--- | :--- | :--- |")
            for idx, op in enumerate(cleaning_history, 1):
                md.append(f"| {idx} | {op.get('operation')} | `{op.get('column')}` | {op.get('rows_affected')} | {op.get('summary')} |")
            md.append("")

        # 6. Notes & Methodology
        md.append("## 4. Methodology & Limitations")
        md.append(quality_score.get("methodology", ""))
        md.append("\n*Report produced deterministically by DataDoctor AI engine without data leakage.*")

        return "\n".join(md)

    @staticmethod
    def export_dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
        """Converts DataFrame to UTF-8 CSV bytes."""
        return df.to_csv(index=False).encode("utf-8")

    @staticmethod
    def export_dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
        """Converts DataFrame to Excel (.xlsx) bytes."""
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Cleaned_Data")
        output.seek(0)
        return output.getvalue()
