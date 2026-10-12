"""
Shared UI Styling, Session State Management, and Navigation Components for DataDoctor AI.
Adheres to modern SaaS design: Dark navy sidebar, clean neutral canvas, blue & cyan accents.
"""

import streamlit as st
import pandas as pd
from typing import Optional, Dict, Any
from src.data_profiler import DataProfiler
from src.quality_score import QualityScoreCalculator


def init_session_state():
    """Initializes persistent application session variables across Streamlit reruns."""
    if "original_df" not in st.session_state:
        st.session_state.original_df = None

    if "cleaned_df" not in st.session_state:
        st.session_state.cleaned_df = None

    if "dataset_meta" not in st.session_state:
        st.session_state.dataset_meta = {}

    if "profile_result" not in st.session_state:
        st.session_state.profile_result = None

    if "quality_score" not in st.session_state:
        st.session_state.quality_score = None

    if "cleaning_history" not in st.session_state:
        st.session_state.cleaning_history = []

    if "undo_stack" not in st.session_state:
        st.session_state.undo_stack = []

    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Overview"


def refresh_profiling():
    """Recalculates profiling metrics and quality score on the working dataset."""
    df = st.session_state.cleaned_df
    if df is not None:
        profile = DataProfiler.profile(df)
        score = QualityScoreCalculator.calculate_score(profile)
        st.session_state.profile_result = profile
        st.session_state.quality_score = score
    else:
        st.session_state.profile_result = None
        st.session_state.quality_score = None


def inject_custom_css():
    """Applies modern SaaS styling tokens and UI refinements."""
    st.markdown(
        """
        <style>
        /* Modern Font & General Layout */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Top Header Styling */
        .main-header {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            padding: 1.5rem 2rem;
            border-radius: 12px;
            color: #ffffff;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.15);
        }

        .main-header h1 {
            color: #ffffff;
            font-size: 1.75rem;
            font-weight: 700;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }

        .main-header p {
            color: #94a3b8;
            font-size: 0.95rem;
            margin: 0.35rem 0 0 0;
        }

        /* Metric Cards */
        .doctor-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
            transition: all 0.2s ease-in-out;
        }

        .doctor-card:hover {
            border-color: #0284c7;
            box-shadow: 0 4px 12px -1px rgba(2, 132, 199, 0.12);
        }

        .card-label {
            font-size: 0.825rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748b;
            margin-bottom: 0.4rem;
        }

        .card-value {
            font-size: 1.85rem;
            font-weight: 800;
            color: #0f172a;
            line-height: 1.2;
        }

        .card-subtext {
            font-size: 0.8rem;
            color: #94a3b8;
            margin-top: 0.25rem;
        }

        /* Status Badge */
        .badge {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.25rem 0.6rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .badge-success { background-color: #ecfdf5; color: #059669; border: 1px solid #a7f3d0; }
        .badge-warning { background-color: #fffbeb; color: #d97706; border: 1px solid #fde68a; }
        .badge-danger { background-color: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
        .badge-info { background-color: #f0f9ff; color: #0284c7; border: 1px solid #bae6fd; }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #0b1120;
            border-right: 1px solid #1e293b;
        }

        [data-testid="stSidebar"] .stMarkdown h1, 
        [data-testid="stSidebar"] .stMarkdown h2, 
        [data-testid="stSidebar"] .stMarkdown h3 {
            color: #f8fafc;
        }

        [data-testid="stSidebar"] .stMarkdown p {
            color: #94a3b8;
        }

        /* Streamlit Button Primary Refinement */
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.15s ease;
        }

        /* Tabs Refinement */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 8px 16px;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Renders the persistent dataset status, memory, and navigation controls in sidebar."""
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 0.5rem 0 1rem 0; border-bottom: 1px solid #1e293b; margin-bottom: 1rem;">
                <div style="display: flex; align-items: center; gap: 0.6rem;">
                    <span style="font-size: 1.6rem;">🩺</span>
                    <div>
                        <div style="font-weight: 800; font-size: 1.15rem; color: #f8fafc; letter-spacing: -0.02em;">DataDoctor AI</div>
                        <div style="font-size: 0.75rem; color: #38bdf8; font-weight: 600;">Data Quality & Analytics Agent</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Dataset Status Section
        st.markdown("### 📊 Active Dataset")
        df = st.session_state.cleaned_df
        meta = st.session_state.dataset_meta

        if df is not None:
            filename = meta.get("filename", "Uploaded Dataset")
            sheet_name = meta.get("sheet_name")
            mem_mb = round(meta.get("memory_usage_bytes", 0) / (1024 * 1024), 2)
            sheet_info = f" • Sheet: `{sheet_name}`" if sheet_name else ""

            st.markdown(
                f"""
                <div style="background: #1e293b; border-radius: 8px; padding: 0.85rem; border: 1px solid #334155; margin-bottom: 1rem;">
                    <div style="font-weight: 600; color: #f8fafc; font-size: 0.9rem; word-break: break-all;">📁 {filename}</div>
                    <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 0.25rem;">
                        {len(df):,} rows × {len(df.columns)} columns {sheet_info}<br>
                        RAM: ~{mem_mb} MB
                    </div>
                    <div style="margin-top: 0.5rem;">
                        <span class="badge badge-success">● Active in Memory</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Undo / Reset Operations
            col_u1, col_u2 = st.columns(2)
            with col_u1:
                can_undo = len(st.session_state.undo_stack) > 0
                if st.button("↩️ Undo", disabled=not can_undo, use_container_width=True, help="Revert the most recent cleaning operation"):
                    if st.session_state.undo_stack:
                        prev_df, last_op = st.session_state.undo_stack.pop()
                        st.session_state.cleaned_df = prev_df
                        if st.session_state.cleaning_history:
                            st.session_state.cleaning_history.pop()
                        refresh_profiling()
                        st.toast(f"Reverted: {last_op}", icon="↩️")
                        st.rerun()

            with col_u2:
                if st.button("🔄 Reset", use_container_width=True, help="Reset to original uploaded dataset"):
                    if st.session_state.original_df is not None:
                        st.session_state.cleaned_df = st.session_state.original_df.copy()
                        st.session_state.cleaning_history.clear()
                        st.session_state.undo_stack.clear()
                        refresh_profiling()
                        st.toast("Working copy reset to original dataset.", icon="🔄")
                        st.rerun()

            if st.button("🗑️ Unload Dataset", use_container_width=True):
                st.session_state.original_df = None
                st.session_state.cleaned_df = None
                st.session_state.dataset_meta = {}
                st.session_state.profile_result = None
                st.session_state.quality_score = None
                st.session_state.cleaning_history.clear()
                st.session_state.undo_stack.clear()
                st.rerun()

        else:
            st.info("No active dataset loaded. Upload a CSV or Excel file to get started.")

        st.markdown("---")
        st.markdown(
            """
            <div style="font-size: 0.75rem; color: #64748b; line-height: 1.4;">
                <b>Security Notice</b><br>
                All data is processed strictly in your local memory. No dataset rows are transmitted externally without your explicit consent.
            </div>
            """,
            unsafe_allow_html=True,
        )
