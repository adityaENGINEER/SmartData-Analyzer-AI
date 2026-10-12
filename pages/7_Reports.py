"""
Consolidated Reports & Dataset Export Page for DataDoctor AI.
Allows exporting cleaned datasets (CSV/Excel), markdown diagnostic health reports, and audit trails.
"""

import streamlit as st
import pandas as pd
from src.ui_components import init_session_state, inject_custom_css, render_sidebar
from src.report_generator import ReportGenerator

init_session_state()
inject_custom_css()
render_sidebar()

st.markdown(
    """
    <div class="main-header">
        <h1>📑 Reports & Dataset Export</h1>
        <p>Export analysis-ready cleaned datasets, comprehensive diagnostic reports, and audit trails.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

df = st.session_state.cleaned_df
meta = st.session_state.dataset_meta
profile = st.session_state.profile_result
score_data = st.session_state.quality_score
history = st.session_state.cleaning_history

if df is None:
    st.warning("⚠️ No dataset loaded. Please upload a dataset or load the sample data from the Overview page.")
    st.stop()

col_exp1, col_exp2 = st.columns([1, 1], gap="large")

# ----------------- 1. Dataset Downloads -----------------
with col_exp1:
    st.markdown("### 💾 Export Cleaned Dataset")
    st.caption("Download the current active working dataset after all applied transformations.")

    base_name = meta.get("filename", "dataset").rsplit(".", 1)[0]

    # CSV Download
    csv_bytes = ReportGenerator.export_dataframe_to_csv_bytes(df)
    st.download_button(
        label="📥 Download Cleaned Dataset as CSV",
        data=csv_bytes,
        file_name=f"{base_name}_cleaned.csv",
        mime="text/csv",
        use_container_width=True,
        type="primary",
    )

    # Excel Download
    try:
        excel_bytes = ReportGenerator.export_dataframe_to_excel_bytes(df)
        st.download_button(
            label="📊 Download Cleaned Dataset as Excel (.xlsx)",
            data=excel_bytes,
            file_name=f"{base_name}_cleaned.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    except Exception as e:
        st.caption(f"Excel export unavailable: {e}")

    st.markdown("---")
    st.markdown("##### Transformation Summary")
    st.write(f"- **Original Records:** {st.session_state.original_df.shape[0]:,} rows × {st.session_state.original_df.shape[1]} cols")
    st.write(f"- **Current Working Records:** {df.shape[0]:,} rows × {df.shape[1]} cols")
    st.write(f"- **Transformations Applied:** {len(history)} cleaning step(s)")

# ----------------- 2. Executive Report Generation -----------------
with col_exp2:
    st.markdown("### 📄 Consolidated Quality Report")
    st.caption("Generate an executive-ready Markdown health report including diagnostics, score, and audit trails.")

    report_md = ReportGenerator.generate_markdown_report(
        filename=meta.get("filename", "Uploaded Dataset"),
        profile_data=profile,
        quality_score=score_data,
        cleaning_history=history,
    )

    st.download_button(
        label="📑 Download Full Diagnostic Report (.md)",
        data=report_md.encode("utf-8"),
        file_name=f"{base_name}_quality_report.md",
        mime="text/markdown",
        use_container_width=True,
        type="secondary",
    )

    with st.expander("👁️ Preview Diagnostic Report", expanded=True):
        st.markdown(report_md)
