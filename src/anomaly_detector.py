"""
Interpretable Anomaly & Outlier Detection Module for DataDoctor AI.
Supports IQR, Z-Score, and Scikit-Learn Isolation Forest with suitability checks and visualizations.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go


class AnomalyDetector:
    """Detects statistical outliers and multivariate anomalies on numeric features."""

    @staticmethod
    def detect_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> Dict[str, Any]:
        """Univariate IQR-based anomaly detection."""
        if column not in df.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        series = df[column].dropna()
        if not pd.api.types.is_numeric_dtype(series):
            raise ValueError(f"Column '{column}' must be numeric for IQR detection.")

        if len(series) < 4:
            raise ValueError("At least 4 non-null numeric values are required.")

        q25 = float(series.quantile(0.25))
        q75 = float(series.quantile(0.75))
        iqr = q75 - q25
        lower_bound = q25 - multiplier * iqr
        upper_bound = q75 + multiplier * iqr

        mask = (df[column] < lower_bound) | (df[column] > upper_bound)
        flagged_df = df[mask].copy()

        return {
            "method": "IQR (Interquartile Range)",
            "column": column,
            "lower_bound": round(lower_bound, 3),
            "upper_bound": round(upper_bound, 3),
            "multiplier": multiplier,
            "anomaly_count": int(mask.sum()),
            "anomaly_pct": round((mask.sum() / len(df)) * 100, 2),
            "flagged_indices": list(flagged_df.index),
            "explanation": f"Values outside [{lower_bound:.2f}, {upper_bound:.2f}] based on {multiplier}x IQR (Q1={q25:.2f}, Q3={q75:.2f}).",
        }

    @staticmethod
    def detect_zscore(df: pd.DataFrame, column: str, threshold: float = 3.0) -> Dict[str, Any]:
        """Univariate Z-Score anomaly detection."""
        if column not in df.columns:
            raise ValueError(f"Column '{column}' does not exist.")

        series = df[column].dropna()
        if not pd.api.types.is_numeric_dtype(series):
            raise ValueError(f"Column '{column}' must be numeric for Z-Score detection.")

        if len(series) < 10:
            raise ValueError("At least 10 observations are recommended for reliable Z-score detection.")

        mean = float(series.mean())
        std = float(series.std())
        if std == 0:
            raise ValueError("Standard deviation is zero; cannot compute Z-scores.")

        z_scores = (df[column] - mean) / std
        mask = z_scores.abs() > threshold
        flagged_df = df[mask].copy()

        return {
            "method": "Z-Score (Standard Deviations)",
            "column": column,
            "mean": round(mean, 3),
            "std": round(std, 3),
            "threshold": threshold,
            "lower_bound": round(mean - threshold * std, 3),
            "upper_bound": round(mean + threshold * std, 3),
            "anomaly_count": int(mask.sum()),
            "anomaly_pct": round((mask.sum() / len(df)) * 100, 2),
            "flagged_indices": list(flagged_df.index),
            "explanation": f"Observations having |Z| > {threshold} (Mean={mean:.2f}, Std={std:.2f}). Assumes roughly bell-shaped distribution.",
        }

    @staticmethod
    def detect_isolation_forest(
        df: pd.DataFrame,
        features: List[str],
        contamination: float = 0.05,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Multivariate Isolation Forest anomaly detection using Scikit-Learn."""
        valid_features = [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
        if len(valid_features) == 0:
            raise ValueError("Select at least 1 valid numeric column for Isolation Forest.")

        # Impute missing values with median for isolation forest
        clean_subset = df[valid_features].dropna()
        if len(clean_subset) < 20:
            raise ValueError(f"Isolation Forest requires at least 20 complete rows. Found {len(clean_subset)} complete rows.")

        iso = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=100)
        preds = iso.fit_predict(clean_subset)
        anomaly_mask = preds == -1

        flagged_indices = clean_subset.index[anomaly_mask].tolist()

        return {
            "method": "Isolation Forest (Unsupervised ML)",
            "features": valid_features,
            "contamination": contamination,
            "anomaly_count": len(flagged_indices),
            "anomaly_pct": round((len(flagged_indices) / len(df)) * 100, 2),
            "flagged_indices": flagged_indices,
            "explanation": f"Trained an ensemble of 100 Isolation Trees on {len(valid_features)} feature(s). Isolates anomalies using recursive feature partitioning.",
        }

    @staticmethod
    def build_anomaly_scatter(
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        flagged_indices: List[Any],
        title: str = "Anomaly Scatter Plot",
    ) -> go.Figure:
        """Visualizes flagged anomalies in an interactive 2D scatter plot."""
        plot_df = df.copy()
        plot_df["Status"] = "Normal"
        plot_df.loc[plot_df.index.isin(flagged_indices), "Status"] = "Anomaly"

        fig = px.scatter(
            plot_df,
            x=x_col,
            y=y_col,
            color="Status",
            color_discrete_map={"Normal": "#0284c7", "Anomaly": "#ef4444"},
            title=title,
            template="plotly_white",
            symbol="Status",
            symbol_map={"Normal": "circle", "Anomaly": "diamond"},
        )
        fig.update_traces(marker=dict(size=8, opacity=0.8))
        fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
        return fig
