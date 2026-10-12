"""
Statistical & Machine Learning Anomaly Detection Page for DataDoctor AI.
"""

import streamlit as st
import pandas as pd
from src.ui_components import init_session_state, inject_custom_css, render_sidebar
from src.anomaly_detector import AnomalyDetector

init_session_state()
inject_custom_css()
render_sidebar()

st.markdown(
    """
    <div class="main-header">
        <h1>🎯 Anomaly Detection Studio</h1>
        <p>Detect outliers with IQR, Z-Score, and Scikit-Learn Isolation Forest algorithms.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

df = st.session_state.cleaned_df
if df is None:
    st.warning("⚠️ No dataset loaded. Please upload a dataset or load the sample data from the Overview page.")
    st.stop()

num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
if not num_cols:
    st.error("Anomaly detection requires at least one numeric column. None found in current dataset.")
    st.stop()

tab_iqr, tab_zscore, tab_iforest = st.tabs(["1. Univariate IQR Method", "2. Univariate Z-Score Method", "3. Multivariate Isolation Forest (ML)"])

# ----------------- 1. IQR Method -----------------
with tab_iqr:
    st.markdown("#### Interquartile Range (IQR) Outlier Detection")
    st.caption("Robust non-parametric method suitable for skewed numerical distributions.")

    col_iqr1, col_iqr2 = st.columns([1, 2])
    with col_iqr1:
        target_num_iqr = st.selectbox("Select numeric feature:", num_cols, key="iqr_col")
        multiplier = st.slider("IQR Multiplier (Default 1.5x):", 1.0, 3.0, 1.5, 0.1)

        run_iqr = st.button("Run IQR Detection", type="primary", key="btn_run_iqr")

    with col_iqr2:
        if run_iqr or target_num_iqr:
            try:
                res_iqr = AnomalyDetector.detect_iqr(df, target_num_iqr, multiplier=multiplier)
                count = res_iqr["anomaly_count"]
                pct = res_iqr["anomaly_pct"]

                st.markdown(
                    f"""
                    <div style="background: #f8fafc; border-left: 4px solid #ef4444; padding: 0.75rem 1rem; border-radius: 6px; margin-bottom: 1rem;">
                        <b>Flagged Anomalies:</b> {count:,} records ({pct}% of rows)<br>
                        <span style="font-size: 0.85rem; color: #64748b;">Valid Bounds: [{res_iqr['lower_bound']:.2f}, {res_iqr['upper_bound']:.2f}]</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if count > 0:
                    st.write("##### Flagged Records Preview:")
                    flagged_df = df.loc[res_iqr["flagged_indices"]]
                    st.dataframe(flagged_df.head(20), use_container_width=True)

                    if len(num_cols) >= 2:
                        y_feature = [c for c in num_cols if c != target_num_iqr][0]
                        fig_scatter = AnomalyDetector.build_anomaly_scatter(
                            df,
                            x_col=target_num_iqr,
                            y_col=y_feature,
                            flagged_indices=res_iqr["flagged_indices"],
                            title=f"IQR Anomalies ({target_num_iqr} vs {y_feature})",
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
            except Exception as e:
                st.error(f"Detection failed: {e}")

# ----------------- 2. Z-Score Method -----------------
with tab_zscore:
    st.markdown("#### Z-Score Standard Deviation Detection")
    st.caption("Assumes roughly normal/bell-shaped distributions. Identifies values far from the mean.")

    col_z1, col_z2 = st.columns([1, 2])
    with col_z1:
        target_num_z = st.selectbox("Select numeric feature:", num_cols, key="z_col")
        z_thresh = st.slider("Z-Score Threshold (Standard deviations):", 2.0, 4.0, 3.0, 0.1)
        run_z = st.button("Run Z-Score Detection", type="primary", key="btn_run_z")

    with col_z2:
        if run_z or target_num_z:
            try:
                res_z = AnomalyDetector.detect_zscore(df, target_num_z, threshold=z_thresh)
                count = res_z["anomaly_count"]
                pct = res_z["anomaly_pct"]

                st.markdown(
                    f"""
                    <div style="background: #f8fafc; border-left: 4px solid #ef4444; padding: 0.75rem 1rem; border-radius: 6px; margin-bottom: 1rem;">
                        <b>Flagged Anomalies:</b> {count:,} records ({pct}% of rows)<br>
                        <span style="font-size: 0.85rem; color: #64748b;">Mean: {res_z['mean']:.2f} • Std: {res_z['std']:.2f} • Bounds: [{res_z['lower_bound']:.2f}, {res_z['upper_bound']:.2f}]</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if count > 0:
                    st.write("##### Flagged Records Preview:")
                    flagged_df = df.loc[res_z["flagged_indices"]]
                    st.dataframe(flagged_df.head(20), use_container_width=True)
            except Exception as e:
                st.error(f"Detection failed: {e}")

# ----------------- 3. Isolation Forest Method -----------------
with tab_iforest:
    st.markdown("#### Scikit-Learn Isolation Forest (Multivariate)")
    st.caption("Ensemble algorithm that partitions multidimensional data into trees to isolate unusual combinations.")

    col_if1, col_if2 = st.columns([1, 2])
    with col_if1:
        selected_features = st.multiselect("Select Features for Model Training:", num_cols, default=num_cols[: min(4, len(num_cols))])
        contamination = st.slider("Estimated Anomaly Contamination (%):", 0.01, 0.20, 0.05, 0.01)

        run_iforest = st.button("Train Isolation Forest Model", type="primary", key="btn_run_if")

    with col_if2:
        if run_iforest:
            try:
                with st.spinner("Training Isolation Forest ensemble..."):
                    res_if = AnomalyDetector.detect_isolation_forest(df, selected_features, contamination=contamination)
                    count = res_if["anomaly_count"]
                    pct = res_if["anomaly_pct"]

                    st.success(f"Model trained successfully on {len(res_if['features'])} feature(s).")
                    st.markdown(
                        f"""
                        <div style="background: #f8fafc; border-left: 4px solid #ef4444; padding: 0.75rem 1rem; border-radius: 6px; margin-bottom: 1rem;">
                            <b>Multivariate Anomalies Detected:</b> {count:,} records ({pct}% of dataset)<br>
                            <span style="font-size: 0.85rem; color: #64748b;">{res_if['explanation']}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if count > 0:
                        st.write("##### Top Flagged Multivariate Anomalies:")
                        flagged_df = df.loc[res_if["flagged_indices"]]
                        st.dataframe(flagged_df.head(20), use_container_width=True)

                        if len(selected_features) >= 2:
                            fig_if_scatter = AnomalyDetector.build_anomaly_scatter(
                                df,
                                x_col=selected_features[0],
                                y_col=selected_features[1],
                                flagged_indices=res_if["flagged_indices"],
                                title=f"Isolation Forest Anomalies ({selected_features[0]} vs {selected_features[1]})",
                            )
                            st.plotly_chart(fig_if_scatter, use_container_width=True)
            except Exception as e:
                st.error(f"Isolation Forest failed: {e}")
