"""
Local Analytical Engine for Natural Language Data Analytics.
Executes deterministic Pandas analytical operations for queries without calling external APIs or executing dangerous eval().
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class AnalyticsEngine:
    """Interprets analytical requests and maps them to verified, deterministic pandas computations."""

    @staticmethod
    def get_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates comprehensive numerical & categorical summary statistics."""
        num_df = df.select_dtypes(include=[np.number])
        cat_df = df.select_dtypes(exclude=[np.number, "datetime"])

        num_summary = {}
        if not num_df.empty:
            desc = num_df.describe().to_dict()
            for col, stats in desc.items():
                num_summary[col] = {k: round(v, 2) if isinstance(v, (int, float)) else v for k, v in stats.items()}

        cat_summary = {}
        if not cat_df.empty:
            for col in cat_df.columns:
                series = cat_df[col].dropna()
                cat_summary[col] = {
                    "count": int(len(series)),
                    "unique": int(series.nunique()),
                    "top_mode": str(series.mode().iloc[0]) if not series.empty else None,
                    "top_freq": int(series.value_counts().iloc[0]) if not series.empty else 0,
                }

        return {
            "numeric_summary": num_summary,
            "categorical_summary": cat_summary,
            "total_rows": len(df),
            "total_columns": len(df.columns),
        }

    @staticmethod
    def find_strongest_correlations(df: pd.DataFrame, top_n: int = 5) -> List[Dict[str, Any]]:
        """Identifies the pairs of numerical variables with the highest absolute correlation."""
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] < 2:
            return []

        corr_matrix = num_df.corr()
        pairs = []
        cols = list(corr_matrix.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                col1, col2 = cols[i], cols[j]
                val = corr_matrix.iloc[i, j]
                if not np.isnan(val):
                    pairs.append({
                        "col1": col1,
                        "col2": col2,
                        "abs_correlation": round(abs(float(val)), 3),
                        "correlation": round(float(val), 3),
                    })

        pairs.sort(key=lambda x: x["abs_correlation"], reverse=True)
        return pairs[:top_n]

    @staticmethod
    def answer_query(df: pd.DataFrame, query: str) -> Dict[str, Any]:
        """
        Parses common natural language queries and executes deterministic local operations.
        No LLM required; 100% offline, predictable, and verifiable.
        """
        q = query.lower().strip()
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

        # Pattern 1: Missing values breakdown
        if any(k in q for k in ["missing", "null", "empty values", "nan"]):
            missing = df.isnull().sum()
            missing_cols = missing[missing > 0].sort_values(ascending=False)
            if missing_cols.empty:
                return {
                    "matched": True,
                    "question": query,
                    "answer": "There are no missing values in any column of this dataset.",
                    "explanation": "Calculated df.isnull().sum() across all columns.",
                    "data": None,
                }
            result_df = pd.DataFrame({
                "Column": missing_cols.index,
                "Missing Count": missing_cols.values,
                "Percentage (%)": (missing_cols.values / len(df) * 100).round(2),
            })
            return {
                "matched": True,
                "question": query,
                "answer": f"Found {len(missing_cols)} column(s) with missing values. Top column is '{missing_cols.index[0]}' with {missing_cols.iloc[0]} missing records ({round(missing_cols.iloc[0]/len(df)*100, 1)}%).",
                "explanation": "Evaluated column-wise null counts and missingness proportions.",
                "data": result_df,
            }

        # Pattern 2: Strongest correlation
        if any(k in q for k in ["correlation", "correlate", "correlated", "relationship"]):
            top_corr = AnalyticsEngine.find_strongest_correlations(df)
            if not top_corr:
                return {
                    "matched": True,
                    "question": query,
                    "answer": "Cannot compute correlation: dataset has fewer than 2 numeric columns.",
                    "explanation": "Pearson correlation requires at least 2 numerical features.",
                    "data": None,
                }
            best = top_corr[0]
            corr_df = pd.DataFrame(top_corr)
            return {
                "matched": True,
                "question": query,
                "answer": f"Strongest correlation is between '{best['col1']}' and '{best['col2']}' (r = {best['correlation']}).",
                "explanation": "Computed Pearson correlation matrix across all numeric features.",
                "data": corr_df,
            }

        # Pattern 3: Highest / Maximum value
        if any(k in q for k in ["highest", "maximum", "max", "top value", "largest"]):
            # Try to match a specific numeric column
            target_col = None
            for col in num_cols:
                if col.lower() in q:
                    target_col = col
                    break
            if not target_col and num_cols:
                target_col = num_cols[0]

            if target_col:
                max_val = df[target_col].max()
                max_row = df.loc[df[target_col] == max_val].head(1)
                return {
                    "matched": True,
                    "question": query,
                    "answer": f"The highest value for '{target_col}' is {max_val}.",
                    "explanation": f"Evaluated max value for column '{target_col}'.",
                    "data": max_row,
                }

        # Pattern 4: Lowest / Minimum value
        if any(k in q for k in ["lowest", "minimum", "min", "smallest", "bottom"]):
            target_col = None
            for col in num_cols:
                if col.lower() in q:
                    target_col = col
                    break
            if not target_col and num_cols:
                target_col = num_cols[0]

            if target_col:
                min_val = df[target_col].min()
                min_row = df.loc[df[target_col] == min_val].head(1)
                return {
                    "matched": True,
                    "question": query,
                    "answer": f"The lowest value for '{target_col}' is {min_val}.",
                    "explanation": f"Evaluated min value for column '{target_col}'.",
                    "data": min_row,
                }

        # Pattern 5: Average / Mean
        if any(k in q for k in ["average", "mean", "avg"]):
            target_col = None
            for col in num_cols:
                if col.lower() in q:
                    target_col = col
                    break
            if not target_col and num_cols:
                target_col = num_cols[0]

            if target_col:
                avg_val = round(float(df[target_col].mean()), 2)
                median_val = round(float(df[target_col].median()), 2)
                return {
                    "matched": True,
                    "question": query,
                    "answer": f"The average (mean) of '{target_col}' is {avg_val} (Median: {median_val}).",
                    "explanation": f"Calculated mean and median across non-null records in '{target_col}'.",
                    "data": pd.DataFrame({"Metric": ["Mean", "Median", "Standard Dev"], "Value": [avg_val, median_val, round(float(df[target_col].std()), 2)]}),
                }

        # Pattern 6: Total / Sum
        if any(k in q for k in ["total", "sum", "overall"]):
            target_col = None
            for col in num_cols:
                if col.lower() in q:
                    target_col = col
                    break
            if not target_col and num_cols:
                target_col = num_cols[0]

            if target_col:
                total_val = round(float(df[target_col].sum()), 2)
                return {
                    "matched": True,
                    "question": query,
                    "answer": f"The total sum of '{target_col}' is {total_val:,.2f}.",
                    "explanation": f"Summed non-null records in '{target_col}'.",
                    "data": None,
                }

        # Pattern 7: Dataset Overview / Summary / Patterns
        if any(k in q for k in ["summary", "summarize", "pattern", "overview", "describe"]):
            return {
                "matched": True,
                "question": query,
                "answer": f"The dataset contains {len(df):,} rows and {len(df.columns)} columns ({len(num_cols)} numeric, {len(cat_cols)} categorical). Total missing cells: {int(df.isnull().sum().sum())}.",
                "explanation": "Generated overall structural shape and column type summary.",
                "data": pd.DataFrame({
                    "Attribute": ["Rows", "Columns", "Numeric Columns", "Categorical Columns", "Duplicate Rows"],
                    "Value": [len(df), len(df.columns), len(num_cols), len(cat_cols), int(df.duplicated().sum())],
                }),
            }

        # Unmatched query
        return {
            "matched": False,
            "question": query,
            "answer": "This analytical question is either ambiguous or requires specific column references.",
            "explanation": f"Supported local queries include: summary/patterns, missing values, correlations, highest/lowest value of a column, average of a column, or total sum of a column. Available columns: {', '.join(df.columns[:8])}{'...' if len(df.columns) > 8 else ''}",
            "data": None,
        }
