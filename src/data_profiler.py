"""
Comprehensive Data Profiling Engine for DataDoctor AI.
Detects missing values, duplicates, column anomalies, data type issues, whitespace errors,
and numerical distribution flags without corrupting the dataset.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd


@dataclass
class QualityIssue:
    issue_type: str
    column: Optional[str]
    severity: str  # "HIGH", "MEDIUM", "LOW", "INFO"
    count: int
    percentage: float
    description: str
    remediation: str
    is_confirmed: bool  # True if confirmed flaw, False if heuristic warning


class DataProfiler:
    """Profiles datasets and detects structural and statistical data quality issues."""

    @classmethod
    def profile(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs comprehensive profiling on the provided dataframe.
        Returns a dict containing statistics, type breakdowns, and detected issues.
        """
        if df is None:
            raise ValueError("Input dataframe cannot be None.")

        total_rows = len(df)
        total_cols = len(df.columns)

        if total_rows == 0 or total_cols == 0:
            return {
                "total_rows": total_rows,
                "total_cols": total_cols,
                "missing_cells": 0,
                "missing_pct": 0.0,
                "duplicate_rows": 0,
                "duplicate_pct": 0.0,
                "column_stats": {},
                "issues": [
                    asdict(
                        QualityIssue(
                            issue_type="Empty Dataset",
                            column=None,
                            severity="HIGH",
                            count=0,
                            percentage=100.0,
                            description="The dataset has 0 rows or 0 columns.",
                            remediation="Upload a dataset containing valid rows and columns.",
                            is_confirmed=True,
                        )
                    )
                ],
                "numeric_cols": [],
                "categorical_cols": [],
                "datetime_cols": [],
                "constant_cols": [],
            }

        issues: List[QualityIssue] = []

        # 1. Missing cell calculations
        missing_matrix = df.isnull()
        missing_per_col = missing_matrix.sum()
        total_cells = total_rows * total_cols
        total_missing_cells = int(missing_per_col.sum())
        missing_pct = round((total_missing_cells / total_cells) * 100, 2) if total_cells > 0 else 0.0

        # Check for entirely empty rows & columns
        empty_rows = int(missing_matrix.all(axis=1).sum())
        if empty_rows > 0:
            issues.append(
                QualityIssue(
                    issue_type="Empty Rows",
                    column=None,
                    severity="HIGH",
                    count=empty_rows,
                    percentage=round((empty_rows / total_rows) * 100, 2),
                    description=f"{empty_rows} row(s) contain entirely missing / NaN values.",
                    remediation="Drop completely empty rows to prevent inaccurate row counts and calculation bias.",
                    is_confirmed=True,
                )
            )

        empty_cols = [c for c in df.columns if missing_per_col[c] == total_rows]
        for ec in empty_cols:
            issues.append(
                QualityIssue(
                    issue_type="Empty Column",
                    column=ec,
                    severity="HIGH",
                    count=total_rows,
                    percentage=100.0,
                    description=f"Column '{ec}' contains 100% missing values.",
                    remediation="Drop this column as it provides zero variance and zero analytical information.",
                    is_confirmed=True,
                )
            )

        # 2. Duplicate rows
        duplicate_rows_count = int(df.duplicated().sum())
        duplicate_pct = round((duplicate_rows_count / total_rows) * 100, 2)
        if duplicate_rows_count > 0:
            issues.append(
                QualityIssue(
                    issue_type="Duplicate Rows",
                    column=None,
                    severity="MEDIUM" if duplicate_pct < 20 else "HIGH",
                    count=duplicate_rows_count,
                    percentage=duplicate_pct,
                    description=f"{duplicate_rows_count} row(s) ({duplicate_pct}%) are exact duplicate copies across all columns.",
                    remediation="Review and de-duplicate records using Smart Cleaning.",
                    is_confirmed=True,
                )
            )

        # 3. Column-by-column detailed profiling
        numeric_cols = []
        categorical_cols = []
        datetime_cols = []
        constant_cols = []
        column_stats: Dict[str, Any] = {}

        for col in df.columns:
            series = df[col]
            col_missing = int(missing_per_col[col])
            col_missing_pct = round((col_missing / total_rows) * 100, 2)
            n_unique = int(series.nunique(dropna=True))

            # Missingness issues per column
            if col_missing > 0 and col not in empty_cols:
                severity = "HIGH" if col_missing_pct > 40 else ("MEDIUM" if col_missing_pct > 10 else "LOW")
                issues.append(
                    QualityIssue(
                        issue_type="Missing Values",
                        column=col,
                        severity=severity,
                        count=col_missing,
                        percentage=col_missing_pct,
                        description=f"Column '{col}' has {col_missing} missing values ({col_missing_pct}%).",
                        remediation="Impute with mean/median for numerical, mode for categorical, or drop if irrecoverable.",
                        is_confirmed=True,
                    )
                )

            # Constant / near-constant columns
            if n_unique == 1 and col_missing < total_rows:
                constant_cols.append(col)
                issues.append(
                    QualityIssue(
                        issue_type="Constant Column",
                        column=col,
                        severity="MEDIUM",
                        count=total_rows - col_missing,
                        percentage=100.0,
                        description=f"Column '{col}' has only one unique value ('{series.dropna().iloc[0]}').",
                        remediation="Consider dropping this column since zero variance provides no analytical utility.",
                        is_confirmed=True,
                    )
                )
            elif n_unique > 1 and total_rows > 20:
                top_freq = series.value_counts(dropna=True).iloc[0]
                top_freq_pct = (top_freq / (total_rows - col_missing)) * 100 if (total_rows - col_missing) > 0 else 0
                if top_freq_pct >= 98.0:
                    issues.append(
                        QualityIssue(
                            issue_type="Near-Constant Column",
                            column=col,
                            severity="LOW",
                            count=int(top_freq),
                            percentage=round(top_freq_pct, 2),
                            description=f"Column '{col}' has a single dominant value appearing in {round(top_freq_pct, 1)}% of rows.",
                            remediation="Inspect if the rare variations are anomalies or data entry errors.",
                            is_confirmed=False,
                        )
                    )

            # Type classification & specific checks
            dtype_str = str(series.dtype)
            stat_info: Dict[str, Any] = {
                "dtype": dtype_str,
                "missing": col_missing,
                "missing_pct": col_missing_pct,
                "unique": n_unique,
            }

            if pd.api.types.is_numeric_dtype(series):
                numeric_cols.append(col)
                non_nulls = series.dropna()
                if not non_nulls.empty:
                    stat_info.update(
                        {
                            "type": "numeric",
                            "min": float(non_nulls.min()),
                            "max": float(non_nulls.max()),
                            "mean": float(non_nulls.mean()),
                            "std": float(non_nulls.std()) if len(non_nulls) > 1 else 0.0,
                            "median": float(non_nulls.median()),
                            "q25": float(non_nulls.quantile(0.25)),
                            "q75": float(non_nulls.quantile(0.75)),
                        }
                    )
                    # Outlier heuristic using IQR
                    iqr = stat_info["q75"] - stat_info["q25"]
                    if iqr > 0:
                        lower_bound = stat_info["q25"] - 1.5 * iqr
                        upper_bound = stat_info["q75"] + 1.5 * iqr
                        outliers = non_nulls[(non_nulls < lower_bound) | (non_nulls > upper_bound)]
                        outlier_count = len(outliers)
                        if outlier_count > 0:
                            stat_info["outliers_count"] = outlier_count
                            issues.append(
                                QualityIssue(
                                    issue_type="Potential Outliers",
                                    column=col,
                                    severity="LOW",
                                    count=outlier_count,
                                    percentage=round((outlier_count / len(non_nulls)) * 100, 2),
                                    description=f"Column '{col}' has {outlier_count} values outside 1.5x IQR [{round(lower_bound, 2)}, {round(upper_bound, 2)}].",
                                    remediation="Examine via Anomaly Detection module. Avoid blind deletion as valid extreme events may exist.",
                                    is_confirmed=False,
                                )
                            )
            elif pd.api.types.is_datetime64_any_dtype(series):
                datetime_cols.append(col)
                non_nulls = series.dropna()
                stat_info.update(
                    {
                        "type": "datetime",
                        "min": str(non_nulls.min()) if not non_nulls.empty else None,
                        "max": str(non_nulls.max()) if not non_nulls.empty else None,
                    }
                )
            else:
                # String / Object column checks
                categorical_cols.append(col)
                str_non_nulls = series.dropna().astype(str)
                stat_info.update(
                    {
                        "type": "categorical",
                        "top_values": str_non_nulls.value_counts().head(5).to_dict(),
                    }
                )

                if not str_non_nulls.empty:
                    # Check leading/trailing whitespace
                    has_whitespace = str_non_nulls.apply(lambda x: len(x) != len(x.strip()))
                    ws_count = int(has_whitespace.sum())
                    if ws_count > 0:
                        issues.append(
                            QualityIssue(
                                issue_type="Whitespace Inconsistency",
                                column=col,
                                severity="LOW",
                                count=ws_count,
                                percentage=round((ws_count / len(str_non_nulls)) * 100, 2),
                                description=f"{ws_count} entry/entries in '{col}' have un-trimmed leading or trailing spaces.",
                                remediation="Apply text trim in Smart Cleaning to prevent duplicate grouping mismatches.",
                                is_confirmed=True,
                            )
                        )

                    # Check for numbers stored as strings
                    converted_numeric = pd.to_numeric(str_non_nulls, errors="coerce")
                    num_valid_conversions = converted_numeric.notnull().sum()
                    if num_valid_conversions > 0 and (num_valid_conversions / len(str_non_nulls)) >= 0.85:
                        issues.append(
                            QualityIssue(
                                issue_type="Potential Numeric Column as Text",
                                column=col,
                                severity="MEDIUM",
                                count=int(num_valid_conversions),
                                percentage=round((num_valid_conversions / len(str_non_nulls)) * 100, 2),
                                description=f"Column '{col}' is stored as text, but {round((num_valid_conversions / len(str_non_nulls)) * 100, 1)}% of values parse as numbers.",
                                remediation="Convert column type to numeric in Smart Cleaning to enable mathematical calculations.",
                                is_confirmed=False,
                            )
                        )

                    # Check for potential date stored as string
                    # Sample up to 20 non-null values for speed
                    sample_dates = str_non_nulls.head(20)
                    converted_dates = pd.to_datetime(sample_dates, errors="coerce", format="mixed")
                    if converted_dates.notnull().sum() >= 18 and num_valid_conversions < len(sample_dates) * 0.5:
                        issues.append(
                            QualityIssue(
                                issue_type="Potential Date as Text",
                                column=col,
                                severity="LOW",
                                count=int(converted_dates.notnull().sum()),
                                percentage=round((converted_dates.notnull().sum() / len(sample_dates)) * 100, 2),
                                description=f"Sample values in '{col}' resemble date/time formats (e.g., '{str_non_nulls.iloc[0]}').",
                                remediation="Convert to datetime in Smart Cleaning to unlock time-series analytics.",
                                is_confirmed=False,
                            )
                        )

            column_stats[col] = stat_info

        return {
            "total_rows": total_rows,
            "total_cols": total_cols,
            "missing_cells": total_missing_cells,
            "missing_pct": missing_pct,
            "duplicate_rows": duplicate_rows_count,
            "duplicate_pct": duplicate_pct,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "datetime_cols": datetime_cols,
            "constant_cols": constant_cols,
            "column_stats": column_stats,
            "issues": [asdict(i) for i in issues],
        }
