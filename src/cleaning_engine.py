"""
Smart Data Cleaning Engine for DataDoctor AI.
Supports previewing, applying, undoing, and auditing non-destructive transformations.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np


class CleaningEngine:
    """Manages cleaning operations on DataFrame copies with preview & audit support."""

    @staticmethod
    def preview_impute_missing(
        df: pd.DataFrame,
        column: str,
        strategy: str,
        custom_value: Optional[Any] = None,
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Preview missing value imputation for a specified column.
        Strategies: 'mean', 'median', 'mode', 'constant', 'drop_rows'
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in dataframe.")

        series = df[column].copy()
        missing_count = int(series.isnull().sum())
        if missing_count == 0:
            return series, {"applied": False, "reason": "No missing values in column."}

        fill_val = None
        if strategy == "mean":
            if not pd.api.types.is_numeric_dtype(series):
                raise ValueError("Mean imputation only applies to numeric columns.")
            fill_val = float(series.mean())
            new_series = series.fillna(fill_val)
        elif strategy == "median":
            if not pd.api.types.is_numeric_dtype(series):
                raise ValueError("Median imputation only applies to numeric columns.")
            fill_val = float(series.median())
            new_series = series.fillna(fill_val)
        elif strategy == "mode":
            mode_series = series.mode(dropna=True)
            if mode_series.empty:
                raise ValueError("Cannot compute mode for an empty column.")
            fill_val = mode_series.iloc[0]
            new_series = series.fillna(fill_val)
        elif strategy == "constant":
            if custom_value is None or str(custom_value).strip() == "":
                raise ValueError("Custom replacement value cannot be blank.")
            fill_val = custom_value
            new_series = series.fillna(fill_val)
        elif strategy == "drop_rows":
            # For preview in series context, we return rows dropped
            new_series = series.dropna()
            return new_series, {
                "applied": True,
                "strategy": strategy,
                "rows_affected": missing_count,
                "fill_value": None,
                "action": f"Drop {missing_count} rows with missing '{column}'",
            }
        else:
            raise ValueError(f"Unsupported imputation strategy '{strategy}'.")

        return new_series, {
            "applied": True,
            "strategy": strategy,
            "rows_affected": missing_count,
            "fill_value": fill_val,
            "action": f"Imputed {missing_count} missing values in '{column}' with {fill_val}",
        }

    @staticmethod
    def apply_impute_missing(
        df: pd.DataFrame,
        column: str,
        strategy: str,
        custom_value: Optional[Any] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Applies imputation and returns a new DataFrame copy with audit info."""
        df_clean = df.copy()
        if strategy == "drop_rows":
            initial_count = len(df_clean)
            df_clean = df_clean.dropna(subset=[column])
            dropped_count = initial_count - len(df_clean)
            return df_clean, {
                "operation": "Drop Missing Rows",
                "column": column,
                "rows_affected": dropped_count,
                "summary": f"Dropped {dropped_count} row(s) containing missing values in '{column}'.",
            }

        new_series, audit = CleaningEngine.preview_impute_missing(df_clean, column, strategy, custom_value)
        df_clean[column] = new_series
        return df_clean, {
            "operation": f"Impute Missing ({strategy.capitalize()})",
            "column": column,
            "rows_affected": audit.get("rows_affected", 0),
            "summary": audit.get("action", ""),
        }

    @staticmethod
    def apply_remove_duplicates(
        df: pd.DataFrame,
        subset_cols: Optional[List[str]] = None,
        keep: str = "first",
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Removes duplicate rows based on all or specified columns."""
        df_clean = df.copy()
        cols_to_check = subset_cols if subset_cols and len(subset_cols) > 0 else None
        initial_len = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=cols_to_check, keep=keep)
        removed_count = initial_len - len(df_clean)

        col_desc = "all columns" if not cols_to_check else f"columns {cols_to_check}"
        return df_clean, {
            "operation": "Remove Duplicate Rows",
            "column": ", ".join(cols_to_check) if cols_to_check else "All",
            "rows_affected": removed_count,
            "summary": f"Removed {removed_count} duplicate row(s) based on {col_desc} (keeping '{keep}').",
        }

    @staticmethod
    def apply_drop_columns(df: pd.DataFrame, columns_to_drop: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Drops selected columns safely."""
        df_clean = df.copy()
        existing = [c for c in columns_to_drop if c in df_clean.columns]
        if not existing:
            raise ValueError("None of the specified columns exist in dataset.")

        df_clean = df_clean.drop(columns=existing)
        return df_clean, {
            "operation": "Drop Columns",
            "column": ", ".join(existing),
            "rows_affected": len(df_clean),
            "summary": f"Dropped {len(existing)} column(s): {', '.join(existing)}.",
        }

    @staticmethod
    def apply_type_conversion(
        df: pd.DataFrame,
        column: str,
        target_type: str,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Converts column to 'numeric', 'datetime', 'string', or 'category'.
        Surfaces conversion failures safely without silent corruption.
        """
        df_clean = df.copy()
        series = df_clean[column]

        if target_type == "numeric":
            converted = pd.to_numeric(series, errors="coerce")
            failures = int((series.notnull() & converted.isnull()).sum())
            df_clean[column] = converted
            return df_clean, {
                "operation": "Convert Type (Numeric)",
                "column": column,
                "rows_affected": len(converted),
                "summary": f"Converted '{column}' to numeric. {failures} unparseable value(s) converted to NaN.",
            }

        elif target_type == "datetime":
            converted = pd.to_datetime(series, errors="coerce", format="mixed")
            failures = int((series.notnull() & converted.isnull()).sum())
            df_clean[column] = converted
            return df_clean, {
                "operation": "Convert Type (DateTime)",
                "column": column,
                "rows_affected": len(converted),
                "summary": f"Converted '{column}' to datetime. {failures} unparseable value(s) converted to NaT.",
            }

        elif target_type == "category":
            df_clean[column] = series.astype("category")
            return df_clean, {
                "operation": "Convert Type (Category)",
                "column": column,
                "rows_affected": len(series),
                "summary": f"Converted '{column}' to categorical type ({series.nunique()} unique categories).",
            }

        elif target_type == "string":
            df_clean[column] = series.astype(str)
            return df_clean, {
                "operation": "Convert Type (String)",
                "column": column,
                "rows_affected": len(series),
                "summary": f"Converted '{column}' to standard string format.",
            }
        else:
            raise ValueError(f"Unsupported target type: {target_type}")

    @staticmethod
    def apply_text_clean(
        df: pd.DataFrame,
        column: str,
        strip_whitespace: bool = True,
        case_normalization: Optional[str] = None,  # "lower", "upper", "title", None
        normalize_spaces: bool = False,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Cleans text columns: trims whitespace, normalizes multiple spaces, adjusts casing."""
        df_clean = df.copy()
        series = df_clean[column].astype(str)

        actions = []
        if strip_whitespace:
            series = series.str.strip()
            actions.append("trimmed whitespace")

        if normalize_spaces:
            series = series.replace(r"\s+", " ", regex=True)
            actions.append("normalized multiple spaces")

        if case_normalization == "lower":
            series = series.str.lower()
            actions.append("converted to lowercase")
        elif case_normalization == "upper":
            series = series.str.upper()
            actions.append("converted to uppercase")
        elif case_normalization == "title":
            series = series.str.title()
            actions.append("converted to title case")

        df_clean[column] = series
        return df_clean, {
            "operation": "Text Cleaning",
            "column": column,
            "rows_affected": len(series),
            "summary": f"Text cleaning applied to '{column}': {', '.join(actions)}.",
        }

    @staticmethod
    def apply_outlier_handling(
        df: pd.DataFrame,
        column: str,
        method: str = "iqr",
        action: str = "cap",  # "cap" (winsorize) or "remove"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Handles statistical outliers using IQR or Z-score capping or dropping."""
        df_clean = df.copy()
        series = df_clean[column]

        if not pd.api.types.is_numeric_dtype(series):
            raise ValueError("Outlier handling requires a numeric column.")

        non_nulls = series.dropna()
        if len(non_nulls) < 4:
            raise ValueError("Insufficient numeric observations to calculate outliers.")

        if method == "iqr":
            q25 = non_nulls.quantile(0.25)
            q75 = non_nulls.quantile(0.75)
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
        elif method == "zscore":
            mean = non_nulls.mean()
            std = non_nulls.std()
            if std == 0:
                raise ValueError("Standard deviation is zero; cannot compute Z-scores.")
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
        else:
            raise ValueError("Method must be 'iqr' or 'zscore'.")

        flagged_mask = (series < lower_bound) | (series > upper_bound)
        flagged_count = int(flagged_mask.sum())

        if flagged_count == 0:
            return df_clean, {
                "operation": f"Outlier Handling ({method.upper()})",
                "column": column,
                "rows_affected": 0,
                "summary": f"No outliers detected in '{column}' using bounds [{lower_bound:.2f}, {upper_bound:.2f}].",
            }

        if action == "cap":
            df_clean[column] = series.clip(lower=lower_bound, upper=upper_bound)
            desc = f"Capped {flagged_count} outlier value(s) in '{column}' to range [{lower_bound:.2f}, {upper_bound:.2f}]."
        elif action == "remove":
            df_clean = df_clean[~flagged_mask]
            desc = f"Removed {flagged_count} outlier row(s) from '{column}' outside [{lower_bound:.2f}, {upper_bound:.2f}]."
        else:
            raise ValueError("Action must be 'cap' or 'remove'.")

        return df_clean, {
            "operation": f"Outlier Handling ({action.capitalize()})",
            "column": column,
            "rows_affected": flagged_count,
            "summary": desc,
        }
