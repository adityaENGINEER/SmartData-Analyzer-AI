"""
Interactive Data Visualizations Page for DataDoctor AI.
Generates responsive Plotly charts tailored to dataset dimensions and data types.
"""

import streamlit as st
import pandas as pd
from src.ui_components import init_session_state, inject_custom_css, render_sidebar
from src.visualization_engine import VisualizationEngine

init_session_state()
inject_custom_css()
render_sidebar()

st.markdown(
    """
    <div class="main-header">
        <h1>📊 Interactive Visualizations</h1>
        <p>Explore distributions, correlations, trends, and category comparisons with Plotly.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

df = st.session_state.cleaned_df
if df is None:
    st.warning("⚠️ No dataset loaded. Please upload a dataset or load the sample data from the Overview page.")
    st.stop()

# 1. Recommended Visualizations Carousel / Dropdown
recommendations = VisualizationEngine.recommend_charts(df)
if recommendations:
    st.markdown("### 💡 Recommended Views for Your Data")
    rec_cols = st.columns(min(len(recommendations), 3))
    for idx, rec in enumerate(recommendations[:3]):
        with rec_cols[idx]:
            st.markdown(f"**{rec['label']}**")
            try:
                fig_rec = VisualizationEngine.build_custom_chart(
                    df,
                    chart_type=rec["chart_type"],
                    x_col=rec["x"],
                    y_col=rec["y"],
                )
                st.plotly_chart(fig_rec, use_container_width=True)
            except Exception as e:
                st.caption(f"Could not render: {e}")

st.markdown("---")

# 2. Custom Visualizations Builder
st.markdown("### 🛠️ Custom Chart Builder")

col_cfg, col_display = st.columns([1, 2.5], gap="medium")

with col_cfg:
    st.markdown("##### Configuration")
    chart_types = ["bar", "line", "scatter", "histogram", "box", "pie", "heatmap"]
    selected_type = st.selectbox("Chart Type:", chart_types, format_func=lambda t: {
        "bar": "Bar Chart",
        "line": "Line Chart (Trend)",
        "scatter": "Scatter Plot",
        "histogram": "Histogram (Distribution)",
        "box": "Box Plot (Spread & Outliers)",
        "pie": "Pie Chart (Proportions)",
        "heatmap": "Correlation Heatmap",
    }.get(t, t))

    x_col = None
    y_col = None
    color_col = None
    agg_func = "sum"

    if selected_type != "heatmap":
        x_col = st.selectbox("X-Axis Feature:", df.columns, key="viz_x")

        if selected_type in ["bar", "line", "scatter", "box"]:
            numeric_options = ["None"] + [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            selected_y = st.selectbox("Y-Axis Feature (Numerical):", numeric_options, key="viz_y")
            y_col = selected_y if selected_y != "None" else None

            if selected_type == "bar" and y_col:
                agg_func = st.selectbox("Aggregation:", ["sum", "mean", "count", "max", "min"])

        cat_options = ["None"] + [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        selected_color = st.selectbox("Group By / Color (Optional):", cat_options, key="viz_color")
        color_col = selected_color if selected_color != "None" else None

with col_display:
    st.markdown("##### Chart Preview")
    try:
        fig_custom = VisualizationEngine.build_custom_chart(
            df,
            chart_type=selected_type,
            x_col=x_col,
            y_col=y_col,
            color_col=color_col,
            agg_func=agg_func,
        )
        st.plotly_chart(fig_custom, use_container_width=True)
    except Exception as e:
        st.error(f"Error rendering chart: {str(e)}")
