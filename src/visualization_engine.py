"""
Interactive Visualization Engine for DataDoctor AI.
Generates responsive, accessible, professionally styled Plotly charts based on column data types.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Modern SaaS Color Palette
CHART_THEME = "plotly_white"
PRIMARY_COLOR = "#0284c7"  # Sky blue
SECONDARY_COLOR = "#0ea5e9"
ACCENT_COLOR = "#06b6d4"  # Cyan
ALERT_COLOR = "#ef4444"
PALETTE = ["#0284c7", "#06b6d4", "#3b82f6", "#6366f1", "#8b5cf6", "#ec4899", "#10b981", "#f59e0b"]


class VisualizationEngine:
    """Creates polished Plotly figures tailored to tabular dataset analysis."""

    @staticmethod
    def recommend_charts(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Recommends appropriate chart types based on dataset column combinations."""
        recs = []
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_datetime64_any_dtype(df[c])]
        dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

        if cat_cols and numeric_cols:
            recs.append({
                "chart_type": "bar",
                "label": f"Category Comparison: {cat_cols[0]} vs {numeric_cols[0]}",
                "x": cat_cols[0],
                "y": numeric_cols[0],
            })

        if dt_cols and numeric_cols:
            recs.append({
                "chart_type": "line",
                "label": f"Time Series Trend: {numeric_cols[0]} over {dt_cols[0]}",
                "x": dt_cols[0],
                "y": numeric_cols[0],
            })

        if len(numeric_cols) >= 2:
            recs.append({
                "chart_type": "scatter",
                "label": f"Correlation Scatter: {numeric_cols[0]} vs {numeric_cols[1]}",
                "x": numeric_cols[0],
                "y": numeric_cols[1],
            })

        if numeric_cols:
            recs.append({
                "chart_type": "histogram",
                "label": f"Distribution: {numeric_cols[0]}",
                "x": numeric_cols[0],
                "y": None,
            })
            recs.append({
                "chart_type": "box",
                "label": f"Outlier Box Plot: {numeric_cols[0]}",
                "x": numeric_cols[0],
                "y": None,
            })

        if len(numeric_cols) >= 2:
            recs.append({
                "chart_type": "heatmap",
                "label": "Correlation Heatmap",
                "x": None,
                "y": None,
            })

        return recs

    @staticmethod
    def create_missingness_chart(df: pd.DataFrame) -> go.Figure:
        """Visualizes missing value percentage per column."""
        missing = df.isnull().sum()
        pct = (missing / len(df)) * 100
        missing_df = pd.DataFrame({"Column": df.columns, "Missing_Count": missing, "Percentage": pct})
        missing_df = missing_df.sort_values(by="Percentage", ascending=True)

        fig = px.bar(
            missing_df,
            x="Percentage",
            y="Column",
            orientation="h",
            title="Missing Values by Column (%)",
            labels={"Percentage": "Missing Percentage (%)", "Column": "Column Name"},
            color="Percentage",
            color_continuous_scale="Reds",
            template=CHART_THEME,
        )
        fig.update_layout(height=max(350, len(df.columns) * 25), margin=dict(l=40, r=40, t=50, b=40))
        return fig

    @staticmethod
    def create_correlation_heatmap(df: pd.DataFrame) -> Optional[go.Figure]:
        """Creates a correlation matrix heatmap for numeric features."""
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return None

        corr = numeric_df.corr().round(2)
        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Pearson Correlation Matrix",
            template=CHART_THEME,
        )
        fig.update_layout(height=480, margin=dict(l=40, r=40, t=50, b=40))
        return fig

    @staticmethod
    def build_custom_chart(
        df: pd.DataFrame,
        chart_type: str,
        x_col: Optional[str] = None,
        y_col: Optional[str] = None,
        color_col: Optional[str] = None,
        agg_func: str = "sum",
    ) -> go.Figure:
        """Builds interactive Plotly charts with error boundaries."""
        if chart_type == "bar":
            if y_col:
                # Group and aggregate if categories have duplicates
                grouped = df.groupby(x_col, as_index=False)[y_col].agg(agg_func).sort_values(by=y_col, ascending=False)
                # Cap to top 25 categories for readability
                grouped = grouped.head(25)
                fig = px.bar(
                    grouped,
                    x=x_col,
                    y=y_col,
                    title=f"{y_col} by {x_col} ({agg_func.capitalize()})",
                    color_discrete_sequence=[PRIMARY_COLOR],
                    template=CHART_THEME,
                )
            else:
                counts = df[x_col].value_counts().head(25).reset_index()
                counts.columns = [x_col, "Count"]
                fig = px.bar(
                    counts,
                    x=x_col,
                    y="Count",
                    title=f"Frequency Count: {x_col}",
                    color_discrete_sequence=[PRIMARY_COLOR],
                    template=CHART_THEME,
                )

        elif chart_type == "line":
            sorted_df = df.sort_values(by=x_col) if x_col else df
            fig = px.line(
                sorted_df,
                x=x_col,
                y=y_col,
                color=color_col,
                title=f"{y_col} Trend over {x_col}",
                template=CHART_THEME,
                color_discrete_sequence=PALETTE,
            )

        elif chart_type == "scatter":
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color=color_col,
                trendline="ols" if (len(df) > 5 and len(df) < 5000 and not color_col) else None,
                title=f"{y_col} vs {x_col}",
                template=CHART_THEME,
                color_discrete_sequence=PALETTE,
            )

        elif chart_type == "histogram":
            fig = px.histogram(
                df,
                x=x_col,
                color=color_col,
                marginal="box",
                title=f"Distribution of {x_col}",
                template=CHART_THEME,
                color_discrete_sequence=PALETTE,
            )

        elif chart_type == "box":
            fig = px.box(
                df,
                x=x_col if y_col else None,
                y=y_col if y_col else x_col,
                color=color_col,
                points="outliers",
                title=f"Box Plot: {y_col if y_col else x_col}",
                template=CHART_THEME,
                color_discrete_sequence=PALETTE,
            )

        elif chart_type == "pie":
            counts = df[x_col].value_counts().head(10).reset_index()
            counts.columns = [x_col, "Count"]
            fig = px.pie(
                counts,
                names=x_col,
                values="Count",
                title=f"Top Categories Distribution: {x_col}",
                template=CHART_THEME,
                color_discrete_sequence=PALETTE,
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")

        elif chart_type == "heatmap":
            fig = VisualizationEngine.create_correlation_heatmap(df)
            if fig is None:
                raise ValueError("At least 2 numeric columns are required to generate a correlation heatmap.")
            return fig
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")

        fig.update_layout(
            margin=dict(l=40, r=40, t=50, b=40),
            hoverlabel=dict(bgcolor="white", font_size=12),
        )
        return fig
