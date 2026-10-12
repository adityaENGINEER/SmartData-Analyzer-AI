"""
Data Quality Engine & Explainable Score Page for DataDoctor AI.
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
        <h1>🔍 Data Quality Engine & Score Breakdown</h1>
        <p>Granular diagnostics, deterministic deductions, and explainable health metrics.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

df = st.session_state.cleaned_df
profile = st.session_state.profile_result
score_data = st.session_state.quality_score

if df is None or profile is None or score_data is None:
    st.warning("⚠️ No dataset currently loaded. Please upload a dataset or load the sample data from the Overview page.")
    st.stop()

# 1. Quality Score Header & Breakdown
col_score_card, col_score_exp = st.columns([1, 2], gap="large")

with col_score_card:
    score = score_data.get("overall_score", 0.0)
    grade = score_data.get("grade", "N/A")
    st.markdown(
        f"""
        <div class="doctor-card" style="text-align: center; padding: 2rem 1.5rem; border: 2px solid #0284c7;">
            <div class="card-label">Overall Health Score</div>
            <div style="font-size: 3.5rem; font-weight: 800; color: #0284c7; line-height: 1.1;">{score}</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #0f172a; margin: 0.5rem 0;">Grade: {grade}</div>
            <div style="font-size: 0.8rem; color: #64748b;">Deterministic Heuristic Index (0–100)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_score_exp:
    st.markdown("#### 📐 Component Breakdown")
    comps = score_data.get("components", {})
    comp_cols = st.columns(4)

    labels = {
        "completeness": ("Completeness", "Missingness & empty records"),
        "uniqueness": ("Uniqueness", "Duplicate records"),
        "consistency": ("Consistency", "Constant cols & text formatting"),
        "integrity": ("Integrity", "Structural validity & outliers"),
    }

    for idx, (key, (title, desc)) in enumerate(labels.items()):
        with comp_cols[idx]:
            c_val = comps.get(key, {})
            sc = c_val.get("score", 0.0)
            mx = c_val.get("max", 0.0)
            ded = c_val.get("deduction", 0.0)
            pct = round((sc / mx) * 100) if mx > 0 else 0
            st.metric(label=title, value=f"{sc}/{mx} pts", delta=f"-{ded} pts" if ded > 0 else "Full Score", delta_color="inverse")
            st.caption(desc)

    st.markdown(f"**Methodology:** {score_data.get('methodology')}")

st.markdown("---")

# 2. Top Impairing Issues
top_impacts = score_data.get("top_impacts", [])
if top_impacts:
    st.markdown("### ⚡ Primary Score Deductions")
    col_imp = st.columns(len(top_impacts))
    for idx, imp in enumerate(top_impacts):
        with col_imp[idx]:
            st.error(imp)

# 3. Missing Value Analysis & Visual Chart
st.markdown("### 📉 Missingness Profile")
col_chart, col_missing_table = st.columns([3, 2])

with col_chart:
    fig_missing = VisualizationEngine.create_missingness_chart(df)
    st.plotly_chart(fig_missing, use_container_width=True)

with col_missing_table:
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)
    m_df = pd.DataFrame({"Missing Rows": missing_counts, "Percentage (%)": missing_pct})
    m_df = m_df[m_df["Missing Rows"] > 0].sort_values(by="Missing Rows", ascending=False)
    if not m_df.empty:
        st.dataframe(m_df, use_container_width=True)
    else:
        st.success("Zero missing values in any column!")

st.markdown("---")

# 4. Detected Issues Registry
st.markdown("### 📋 Complete Issues & Recommendations Registry")
issues = profile.get("issues", [])
if not issues:
    st.success("No issues detected.")
else:
    # Filter controls
    severities = ["All"] + sorted(list({i["severity"] for i in issues}))
    sel_sev = st.selectbox("Filter issues by severity:", severities)

    filtered_issues = issues if sel_sev == "All" else [i for i in issues if i["severity"] == sel_sev]

    for iss in filtered_issues:
        sev = iss["severity"]
        border_col = "#ef4444" if sev == "HIGH" else ("#f59e0b" if sev == "MEDIUM" else "#3b82f6")
        confirmed_tag = "Confirmed Flaw" if iss["is_confirmed"] else "Heuristic Warning"
        col_tag = f"Column: <code>{iss['column']}</code>" if iss["column"] else "Whole Dataset"

        st.markdown(
            f"""
            <div style="background: #ffffff; border-left: 5px solid {border_col}; border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: 0.85rem; border-top: 1px solid #e2e8f0; border-right: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                    <div style="font-weight: 700; font-size: 1rem; color: #0f172a;">{iss['issue_type']}</div>
                    <div>
                        <span class="badge badge-info">{confirmed_tag}</span>
                        <span style="font-size: 0.75rem; font-weight: 700; color: {border_col}; margin-left: 0.5rem;">{sev} PRIORITY</span>
                    </div>
                </div>
                <div style="font-size: 0.875rem; color: #475569; margin-bottom: 0.5rem;">
                    {iss['description']} ({iss['count']} affected records, {iss['percentage']}%)
                </div>
                <div style="font-size: 0.825rem; background: #f8fafc; padding: 0.5rem 0.75rem; border-radius: 6px; border: 1px solid #e2e8f0;">
                    <b>💡 Recommended Action:</b> {iss['remediation']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
