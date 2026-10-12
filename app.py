"""
DataDoctor AI — Intelligent Data Quality & Analytics Agent
Main Application Entrypoint
"""

import streamlit as st
import pandas as pd
from src.ui_components import init_session_state, inject_custom_css, render_sidebar, refresh_profiling
from src.data_loader import DataLoader
from src.analytics_engine import AnalyticsEngine
from src.visualization_engine import VisualizationEngine

# Page configuration
st.set_page_config(
    page_title="DataDoctor AI — Data Quality & Analytics Agent",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize application state & styling
init_session_state()
inject_custom_css()
render_sidebar()

# Main Header Banner
st.markdown(
    """
    <div class="main-header">
        <h1>🩺 DataDoctor AI</h1>
        <p>Diagnose Your Data. Fix the Issues. Discover the Insights.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# If no dataset is loaded, render the Hero Upload & Getting Started view
if st.session_state.cleaned_df is None:
    col_intro, col_upload = st.columns([1.1, 1], gap="large")

    with col_intro:
        st.markdown("### Transform Raw Datasets into Analysis-Ready Data")
        st.markdown(
            """
            **DataDoctor AI** is an intelligent data quality and analytics agent designed for data analysts, 
            scientists, and business teams. It detects flaws, executes auditable cleaning, computes explainable 
            quality scores, and generates analytical insights without compromising your data privacy.

            #### 🚀 Core Capabilities
            - **🔍 Automated Diagnostics:** Instantly profile nulls, duplicates, mixed types, and structural outliers.
            - **💯 Explainable Quality Score:** Transparent 0–100 health index with deterministic scoring rules.
            - **✨ Non-Destructive Cleaning:** Preview and undo missing imputation, de-duplication, text trimming, and outlier winsorization.
            - **📈 Interactive Visualizations:** Beautiful Plotly heatmaps, distributions, and scatter trends.
            - **🤖 AI & Local Analyst:** Query your data with natural language offline or using your private LLM key.
            - **⚡ SQL Playground:** In-memory SQLite interface with read-only analytical protection.
            """
        )

        # Quick sample dataset button for immediate user exploration
        st.markdown("#### 🧪 Test With Sample Data")
        if st.button("Load Built-in Sample Dataset (Sales & Employees)", use_container_width=True, type="secondary"):
            sample_data = {
                "Employee_ID": ["EMP-001", "EMP-002", "EMP-003", "EMP-004", "EMP-005", "EMP-006", "EMP-006", "EMP-007", "EMP-008", "EMP-009"],
                "Full_Name": ["Alice Smith", "Bob Jones", "Charlie Brown", "Diana Prince", "Evan Wright", "Evan Wright", "Evan Wright", "Fiona Gallagher", "George Clark", "Hannah Abbott"],
                "Department": ["Sales", "Engineering", "Marketing", "Executive", " Engineering ", "Engineering", "Engineering", "Sales", "Marketing", "Engineering"],
                "Salary": [62000, 85000, 58000, 145000, 82000, 82000, 82000, None, 59000, 91000],
                "Performance_Score": [4.2, 4.8, 3.9, 4.9, 4.1, 4.1, 4.1, 3.5, 4.0, None],
                "Tenure_Years": [3, 5, 2, 8, 4, 4, 4, 1, 3, 6],
                "Join_Date": ["2021-03-15", "2019-06-01", "2022-08-20", "2016-01-10", "2020-11-05", "2020-11-05", "2020-11-05", "2023-04-12", "2021-09-30", "2018-07-14"],
                "Bonus_Amount": ["5000", "7500", "3000", "25000", "6500", "6500", "6500", "2000", "3200", "8000"],
            }
            sample_df = pd.DataFrame(sample_data)
            st.session_state.original_df = sample_df.copy()
            st.session_state.cleaned_df = sample_df.copy()
            st.session_state.dataset_meta = {
                "filename": "sample_employees.csv",
                "row_count": len(sample_df),
                "col_count": len(sample_df.columns),
                "memory_usage_bytes": int(sample_df.memory_usage(deep=True).sum()),
                "sheet_name": None,
            }
            refresh_profiling()
            st.rerun()

    with col_upload:
        st.markdown(
            """
            <div class="doctor-card" style="border: 2px dashed #94a3b8; text-align: center; padding: 2rem 1.5rem;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📤</div>
                <div style="font-weight: 700; font-size: 1.15rem; color: #0f172a;">Upload Your Dataset</div>
                <div style="font-size: 0.85rem; color: #64748b; margin-bottom: 1.5rem;">
                    Drag and drop CSV or Excel (.xlsx, .xls) files up to 100 MB.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            help="Files are processed entirely in local memory.",
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            filename = uploaded_file.name
            file_bytes = uploaded_file.read()

            # Handle multi-sheet Excel
            sheet_to_load = None
            if filename.lower().endswith((".xlsx", ".xls")):
                try:
                    sheets = DataLoader.inspect_excel_sheets(file_bytes)
                    if len(sheets) > 1:
                        sheet_to_load = st.selectbox("Select Excel Worksheet to Analyze:", sheets)
                except Exception as ex:
                    st.error(f"Error inspecting Excel file: {ex}")

            if st.button("🚀 Ingest & Diagnose Dataset", use_container_width=True, type="primary"):
                try:
                    with st.spinner("Parsing file and verifying data integrity..."):
                        df, meta = DataLoader.load_dataset(file_bytes, filename, sheet_name=sheet_to_load)
                        st.session_state.original_df = df.copy()
                        st.session_state.cleaned_df = df.copy()
                        st.session_state.dataset_meta = meta
                        refresh_profiling()
                        st.toast(f"Successfully loaded '{filename}' ({len(df):,} rows)", icon="✅")
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to ingest dataset: {str(e)}")

else:
    # DATASET LOADED: Render the Overview Hub
    df = st.session_state.cleaned_df
    meta = st.session_state.dataset_meta
    profile = st.session_state.profile_result
    score_data = st.session_state.quality_score

    # Metric Row 1: KPI Cards
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

    with col_m1:
        st.markdown(
            f"""
            <div class="doctor-card">
                <div class="card-label">Total Rows</div>
                <div class="card-value">{len(df):,}</div>
                <div class="card-subtext">Active working records</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m2:
        st.markdown(
            f"""
            <div class="doctor-card">
                <div class="card-label">Total Columns</div>
                <div class="card-value">{len(df.columns)}</div>
                <div class="card-subtext">{len(profile.get('numeric_cols', []))} numeric, {len(profile.get('categorical_cols', []))} text</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m3:
        missing_cells = profile.get("missing_cells", 0)
        missing_pct = profile.get("missing_pct", 0.0)
        badge = "badge-danger" if missing_pct > 15 else ("badge-warning" if missing_pct > 0 else "badge-success")
        st.markdown(
            f"""
            <div class="doctor-card">
                <div class="card-label">Missing Values</div>
                <div class="card-value">{missing_cells:,}</div>
                <div class="card-subtext"><span class="badge {badge}">{missing_pct}% of total cells</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m4:
        dup_rows = profile.get("duplicate_rows", 0)
        dup_pct = profile.get("duplicate_pct", 0.0)
        badge = "badge-danger" if dup_pct > 10 else ("badge-warning" if dup_pct > 0 else "badge-success")
        st.markdown(
            f"""
            <div class="doctor-card">
                <div class="card-label">Duplicate Rows</div>
                <div class="card-value">{dup_rows:,}</div>
                <div class="card-subtext"><span class="badge {badge}">{dup_pct}% duplication</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_m5:
        overall_score = score_data.get("overall_score", 0.0)
        grade = score_data.get("grade", "N/A")
        badge = "badge-success" if overall_score >= 80 else ("badge-warning" if overall_score >= 60 else "badge-danger")
        st.markdown(
            f"""
            <div class="doctor-card" style="border-left: 4px solid #0284c7;">
                <div class="card-label">Quality Score</div>
                <div class="card-value" style="color: #0284c7;">{overall_score} <span style="font-size: 1rem; color: #64748b;">/100</span></div>
                <div class="card-subtext"><span class="badge {badge}">{grade}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # Tabs for Quick Dataset Exploration
    tab1, tab2, tab3 = st.tabs(["📋 Dataset Preview & Schema", "📊 Numerical Summary Statistics", "🔍 Quality Diagnostic Highlights"])

    with tab1:
        col_p1, col_p2 = st.columns([3, 1])
        with col_p1:
            st.markdown("##### First 100 Rows")
            st.dataframe(df.head(100), use_container_width=True, height=380)
        with col_p2:
            st.markdown("##### Column Schema")
            schema_data = []
            for c in df.columns:
                schema_data.append({
                    "Column": c,
                    "Type": str(df[c].dtype),
                    "Nulls": int(df[c].isnull().sum()),
                })
            st.dataframe(pd.DataFrame(schema_data), use_container_width=True, height=380)

    with tab2:
        num_summary = AnalyticsEngine.get_summary_statistics(df)["numeric_summary"]
        if num_summary:
            st.dataframe(pd.DataFrame(num_summary).T, use_container_width=True)
        else:
            st.info("No numerical columns available for summary statistics.")

    with tab3:
        issues = profile.get("issues", [])
        if not issues:
            st.success("🎉 No data quality issues detected in this dataset!")
        else:
            st.markdown(f"Found **{len(issues)}** flagged data quality observation(s):")
            issue_df = pd.DataFrame([
                {
                    "Type": i["issue_type"],
                    "Column": i["column"] or "Dataset Level",
                    "Severity": i["severity"],
                    "Affected Records": f"{i['count']:,} ({i['percentage']}%)",
                    "Remediation": i["remediation"],
                }
                for i in issues
            ])
            st.dataframe(issue_df, use_container_width=True)
            st.info("💡 Navigate to **Data Quality** or **Smart Cleaning** in the sidebar to review and fix these issues.")
